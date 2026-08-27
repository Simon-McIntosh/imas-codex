"""Catalog approval records its editorial outcome without rewriting provenance."""

from __future__ import annotations

from typing import Any

from imas_codex.standard_names.promote import (
    mark_catalog_name_approved,
    resolve_contested_override,
)


class _ApprovalGraph:
    """Stateful graph double for atomic approval and change-row writes."""

    def __init__(self, **properties: Any) -> None:
        self.properties = properties
        self.changes: list[dict[str, Any]] = []
        self.statements: list[str] = []

    def query(self, statement: str, **parameters: Any) -> list[dict[str, str]]:
        self.statements.append(statement)
        contested_override = "name_stage: 'contested'" in statement
        if contested_override:
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

        self.properties.setdefault("catalog_approved_at", "approval-time")
        if "CREATE (change:StandardNameChange" in statement:
            self.changes.append(
                {
                    "from_name": parameters["name"],
                    "to_name": parameters["name"],
                    "operation": parameters["editorial_outcome"],
                    "reason": parameters.get("change_reason", parameters.get("reason")),
                    "origin": parameters["change_origin"],
                    "internal": True,
                }
            )
        return [{"id": parameters["name"]}]

    def close(self) -> None:
        pass


def _generation_provenance(graph: _ApprovalGraph) -> tuple[Any, ...]:
    return tuple(
        graph.properties[field]
        for field in ("origin", "model", "generated_at", "chain_length")
    )


def _generated_identity(**properties: Any) -> dict[str, Any]:
    return {
        "origin": "pipeline",
        "model": "configured-generation-seat",
        "generated_at": "2026-07-01T12:34:56Z",
        "chain_length": 3,
        **properties,
    }


def test_approval_rows_distinguish_ratification_from_content_edit() -> None:
    unchanged = _ApprovalGraph(
        **_generated_identity(name_stage="accepted", docs_stage="accepted")
    )
    unchanged_provenance = _generation_provenance(unchanged)

    assert mark_catalog_name_approved(
        "plasma_current",
        catalog_pr_number=42,
        catalog_pr_url="https://example.invalid/pull/42",
        catalog_merge_commit_sha="abc123",
        gc=unchanged,
    )

    assert _generation_provenance(unchanged) == unchanged_provenance
    assert len(unchanged.changes) == 1
    assert unchanged.statements[0].count("CREATE (change:StandardNameChange") == 1
    assert unchanged.statements[0].count("HAS_INTERNAL_CHANGE") == 1
    unchanged_row = unchanged.changes[0]
    assert unchanged_row["operation"] == "unchanged_ratification"
    assert unchanged_row["from_name"] == unchanged_row["to_name"] == "plasma_current"
    assert unchanged_row["origin"] == "catalog_promotion"

    edited = _ApprovalGraph(
        **_generated_identity(name_stage="contested", docs_stage="accepted")
    )
    edited_provenance = _generation_provenance(edited)

    assert resolve_contested_override(
        "plasma_current", reason="Accept the reviewed content edit.", gc=edited
    )

    assert _generation_provenance(edited) == edited_provenance
    assert len(edited.changes) == 1
    assert edited.statements[0].count("CREATE (change:StandardNameChange") == 1
    assert edited.statements[0].count("HAS_INTERNAL_CHANGE") == 1
    edited_row = edited.changes[0]
    assert edited_row["operation"] == "content_edit"
    assert edited_row["from_name"] == edited_row["to_name"] == "plasma_current"
    assert edited_row["origin"] == "catalog_override"
    assert edited_row["operation"] != unchanged_row["operation"]
