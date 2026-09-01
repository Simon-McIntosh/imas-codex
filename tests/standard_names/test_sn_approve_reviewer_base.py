"""Reviewer edits are compared with the catalog content originally submitted."""

from __future__ import annotations

import json
import subprocess
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

from imas_codex.standard_names.promote import (
    ApprovalChange,
    read_pr_changes,
    resolve_merged_pr,
    run_approval,
)


def _git(repo: Path, *args: str) -> str:
    return subprocess.run(
        ["git", *args],
        cwd=repo,
        capture_output=True,
        text=True,
        check=True,
    ).stdout.strip()


def _write_entries(path: Path, *, edited_description: str | None = None) -> None:
    description = edited_description or "The initial duration description."
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        "- name: pulse_duration\n"
        "  kind: scalar\n"
        "  unit: s\n"
        f"  description: {description}\n\n"
        "- name: plasma_current\n"
        "  kind: scalar\n"
        "  unit: A\n"
        "  description: The initial current description.\n",
        encoding="utf-8",
    )


def test_additive_pr_detects_edit_against_cut_time_catalog(tmp_path: Path) -> None:
    repo = tmp_path / "catalog"
    repo.mkdir()
    _git(repo, "init", "-q", "-b", "main")
    _git(repo, "config", "user.email", "review@example.test")
    _git(repo, "config", "user.name", "Catalog Reviewer")
    _git(repo, "commit", "--allow-empty", "-q", "-m", "blank catalog")

    head_ref = "review/v0.3.0rc1+west-task-2e"
    _git(repo, "checkout", "-q", "-b", head_ref)
    catalog_path = repo / "standard_names" / "equilibrium.yml"
    _write_entries(catalog_path)
    _git(repo, "add", "standard_names/equilibrium.yml")
    _git(repo, "commit", "-q", "-m", "submit catalog entries")
    cut_commit = _git(repo, "rev-parse", "HEAD")
    cut_tag = head_ref.removeprefix("review/")
    _git(repo, "tag", "-a", cut_tag, "-m", "catalog review candidate")

    revised = "Reviewer wording after the candidate was cut."
    _write_entries(catalog_path, edited_description=revised)
    _git(repo, "add", "standard_names/equilibrium.yml")
    _git(repo, "commit", "-q", "-m", "revise duration description")
    edit_commit = _git(repo, "rev-parse", "HEAD")

    _git(repo, "checkout", "-q", "main")
    _git(repo, "merge", "-q", "--no-ff", head_ref, "-m", "merge reviewed catalog")
    merge_commit = _git(repo, "rev-parse", "HEAD")
    additive_base = f"{merge_commit}^1"

    assert read_pr_changes(repo, additive_base) == []

    pr_url = "https://github.com/example/catalog/pull/3"
    payload = {
        "number": 3,
        "url": pr_url,
        "state": "MERGED",
        "mergeCommit": {"oid": merge_commit},
        "author": {"login": "reviewer"},
        "headRefName": head_ref,
        "baseRefName": "main",
        "commits": [{"oid": cut_commit}, {"oid": edit_commit}],
    }
    with patch(
        "imas_codex.standard_names.promote.subprocess.run",
        return_value=SimpleNamespace(
            returncode=0,
            stdout=json.dumps(payload),
            stderr="",
        ),
    ):
        resolved = resolve_merged_pr(pr_url)

    assert resolved.review_base_ref == cut_tag
    changes = read_pr_changes(repo, resolved.review_base_ref)
    assert changes == [
        ApprovalChange(
            sn_id="pulse_duration",
            axis="docs",
            old_value="The initial duration description.",
            new_value=revised,
        )
    ]

    with patch("imas_codex.standard_names.promote._name_exists", return_value=True):
        report = run_approval(
            isnc_dir=repo,
            base_ref=additive_base,
            catalog_pr_number=resolved.number,
            catalog_pr_url=resolved.url,
            catalog_merge_commit_sha=resolved.merge_commit,
            batch=["pulse_duration", "plasma_current"],
            dry_run=True,
            gc=object(),
        )

    assert report.changes_seen == 1
    assert [
        (outcome.sn_id, outcome.axis, outcome.decision) for outcome in report.outcomes
    ] == [("pulse_duration", "docs", "planned")]
