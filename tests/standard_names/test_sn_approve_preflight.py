"""Approval eligibility is established before any fold-back mutation."""

from __future__ import annotations

from types import SimpleNamespace
from typing import Any
from unittest.mock import MagicMock, patch

from click.testing import CliRunner

from imas_codex.cli.sn import sn
from imas_codex.standard_names.promote import ApprovalChange, undo_approval


class _IneligibleGraph:
    def __init__(self) -> None:
        self.statements: list[str] = []

    def query(self, statement: str, **parameters: Any) -> list[dict[str, int]]:
        self.statements.append(statement)
        if "APPROVAL_EDIT_ELIGIBILITY" in statement:
            return [{"n": 0}]
        if "APPROVAL_MATCH_BY_ID" in statement:
            return [{"n": 1}]
        raise AssertionError(f"unexpected query: {statement}")

    def close(self) -> None:
        pass


def test_ineligible_edit_exits_before_graph_or_catalog_mutation(tmp_path) -> None:
    artifact = tmp_path / "review.sn_names.yaml"
    artifact.write_text(
        "kind: sn_names\nschema_version: 1\nname: review\nnames:\n- ineligible_entry\n",
        encoding="utf-8",
    )
    change = ApprovalChange(
        sn_id="ineligible_entry",
        axis="docs",
        old_value="Accepted description.",
        new_value="Reviewer-edited description.",
    )
    graph = _IneligibleGraph()

    with (
        patch(
            "imas_codex.standard_names.promote.read_pr_changes",
            return_value=[change],
        ),
        patch(
            "imas_codex.standard_names.promote._prepare_additive_catalog_delta",
            return_value=SimpleNamespace(added_names=frozenset({change.sn_id})),
        ),
        patch(
            "imas_codex.standard_names.promote.GraphClient",
            return_value=graph,
        ),
        patch("imas_codex.standard_names.promote.apply_edit") as apply_edit,
        patch(
            "imas_codex.standard_names.promote.mark_catalog_name_approved"
        ) as auto_promote,
        patch(
            "imas_codex.standard_names.promote._commit_catalog_correction"
        ) as materialize_catalog,
        patch("imas_codex.standard_names.promote.tag_fold_back") as write_tag,
    ):
        result = CliRunner().invoke(
            sn,
            [
                "approve",
                "--isnc",
                str(tmp_path),
                "--base",
                "submitted-candidate",
                "--pr-number",
                "7",
                "--pr-url",
                "https://github.com/example/catalog/pull/7",
                "--merge-commit",
                "merge-seven",
                "--batch",
                str(artifact),
                "--no-notes",
            ],
        )

    assert result.exit_code != 0, result.output
    assert "not approval-eligible" in result.output
    assert graph.statements
    assert all(
        token not in statement.upper()
        for statement in graph.statements
        for token in (" SET ", " CREATE ", " MERGE ", " DELETE ", " REMOVE ")
    )
    apply_edit.assert_not_called()
    auto_promote.assert_not_called()
    materialize_catalog.assert_not_called()
    write_tag.assert_not_called()


def test_undo_restores_docs_stage_for_approved_and_contested_rows() -> None:
    graph = MagicMock()
    graph.query.side_effect = [
        [{"id": "approved_entry"}],
        [{"id": "contested_entry"}],
    ]

    report = undo_approval(
        pr_number=7,
        batch=["approved_entry", "contested_entry"],
        gc=graph,
    )

    assert report.demoted == ["approved_entry"]
    assert report.contested_reverted == ["contested_entry"]
    approved_query = graph.query.call_args_list[0].args[0]
    contested_query = graph.query.call_args_list[1].args[0]
    assert "sn.docs_stage = 'accepted'" in approved_query
    assert "sn.docs_stage = 'accepted'" in contested_query
