"""Reviewer edits are compared with the catalog content originally submitted."""

from __future__ import annotations

import subprocess
from pathlib import Path
from unittest.mock import patch

import pytest

from imas_codex.standard_names.promote import (
    ApprovalChange,
    read_pr_changes,
    resolve_merged_pr,
    run_approval,
)


class RealTransportAttempted(BaseException):
    """A test reached the network instead of its mock.

    Derived from BaseException rather than Exception on purpose: the readers
    under test degrade a failed GitHub call to an empty result through a broad
    ``except Exception``, so an ordinary error would be swallowed and an
    escaped mock would read as a behaviour change. This one propagates.
    """


@pytest.fixture(autouse=True)
def no_real_transport(monkeypatch):
    """Any HTTP request escaping a mock fails the test instead of leaving.

    An unauthenticated GitHub read answers 404, which the resolver reports as
    an ordinary rejection — so a mock that stopped intercepting would look
    like a behaviour change rather than a test that reached the network.
    """

    def refuse(request, *args, **kwargs):
        raise RealTransportAttempted(
            f"test opened a real connection to {getattr(request, 'full_url', request)}"
        )

    monkeypatch.setattr("urllib.request.urlopen", refuse)


def _git(repo: Path, *args: str) -> str:
    return subprocess.check_output(
        ["git", *args],
        cwd=repo,
        text=True,
        stderr=subprocess.DEVNULL,
    ).strip()


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
    cut_tag = head_ref.removeprefix("review/")
    _git(repo, "tag", "-a", cut_tag, "-m", "catalog review candidate")

    revised = "Reviewer wording after the candidate was cut."
    _write_entries(catalog_path, edited_description=revised)
    _git(repo, "add", "standard_names/equilibrium.yml")
    _git(repo, "commit", "-q", "-m", "revise duration description")

    _git(repo, "checkout", "-q", "main")
    _git(repo, "merge", "-q", "--no-ff", head_ref, "-m", "merge reviewed catalog")
    merge_commit = _git(repo, "rev-parse", "HEAD")
    additive_base = f"{merge_commit}^1"

    assert read_pr_changes(repo, additive_base) == []

    pr_url = "https://github.com/example/catalog/pull/3"
    # REST reports a merged pull request as closed with a merge record beside
    # it, so the merged disposition is carried by ``merged``, not by ``state``.
    payload = {
        "number": 3,
        "html_url": pr_url,
        "state": "closed",
        "merged": True,
        "merge_commit_sha": merge_commit,
        "user": {"login": "reviewer"},
        "head": {"ref": head_ref},
        "base": {"ref": "main"},
    }
    with patch(
        "imas_codex.graph.ghcr.github_api_call",
        return_value=(200, payload),
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


def test_guard_stops_a_resolve_whose_mock_misses_the_transport_seam() -> None:
    """A mock aimed away from the REST seam is caught, not quietly tolerated.

    The resolver would otherwise reach GitHub unauthenticated and report the
    404 as an ordinary rejection, so this exercises the autouse guard rather
    than asserting it exists: the mock below intercepts a helper that sits
    *behind* the transport, leaving the transport itself live.
    """
    with (
        patch(
            "imas_codex.standard_names.promote._pull_request_state",
            return_value="MERGED",
        ),
        patch("imas_codex.graph.ghcr.resolve_api_token", return_value="token"),
        pytest.raises(RealTransportAttempted),
    ):
        resolve_merged_pr("https://github.com/example/catalog/pull/3")
