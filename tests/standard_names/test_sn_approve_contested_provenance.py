"""Merged catalog provenance remains attached to contested reviewer edits."""

from __future__ import annotations

from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

from click.testing import CliRunner

from imas_codex.standard_names.promote import (
    ApprovalChange,
    ApprovalReport,
    run_approval,
)


class _RecordingGraph:
    def __init__(self) -> None:
        self.names = {"pulse_duration": {"id": "pulse_duration"}}

    def query(self, query: str, **params: object) -> list[dict[str, object]]:
        if "APPROVAL_MATCH_BY_ID" in query:
            return [{"n": 1}]
        if "APPROVAL_CONTEST" in query:
            node = self.names[str(params["id"])]
            node.update(
                catalog_pr_number=params["pr_number"],
                catalog_pr_url=params["pr_url"],
                catalog_merge_commit_sha=params["merge_commit"],
                catalog_reviewer_actor=params["reviewer_actor"],
            )
        return []


def test_contested_reviewer_edit_retains_merged_pr_provenance() -> None:
    change = ApprovalChange(
        sn_id="pulse_duration",
        axis="docs",
        old_value="Initial pulse duration documentation.",
        new_value="Reviewer-edited pulse duration documentation.",
    )
    graph = _RecordingGraph()
    edit_plan = SimpleNamespace(
        blocked=None,
        successor=None,
        run_id="sn-edit-test",
    )
    provenance = {
        "catalog_pr_number": 3,
        "catalog_pr_url": "https://github.com/example/catalog/pull/3",
        "catalog_merge_commit_sha": "0123456789abcdef",
        "catalog_reviewer_actor": "catalog-reviewer",
    }

    with (
        patch(
            "imas_codex.standard_names.promote.read_pr_changes",
            return_value=[change],
        ),
        patch("imas_codex.standard_names.promote.apply_edit", return_value=edit_plan),
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
    assert {
        field: graph.names["pulse_duration"][field] for field in provenance
    } == provenance


def test_approval_keeps_frozen_review_artifact_byte_exact(tmp_path: Path) -> None:
    from imas_codex.cli.sn import sn

    artifact = tmp_path / "review.sn_names.yaml"
    original = (
        b"kind: sn_names\n"
        b"schema_version: 1\n"
        b"name: review\n"
        b"names:\n"
        b"- pulse_duration\n"
        b"merge_commit: null\n"
    )
    artifact.write_bytes(original)

    with patch(
        "imas_codex.standard_names.promote.run_approval",
        return_value=ApprovalReport(threshold=0.85),
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
                "3",
                "--pr-url",
                "https://github.com/example/catalog/pull/3",
                "--merge-commit",
                "0123456789abcdef",
                "--batch",
                str(artifact),
                "--no-notes",
            ],
        )

    assert result.exit_code == 0, result.output
    assert artifact.read_bytes() == original
