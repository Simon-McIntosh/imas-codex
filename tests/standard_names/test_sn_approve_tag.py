"""Tests for the fold-back receipt — the version tag ``sn approve`` writes.

A fold-back requires a durable receipt linking the merged catalog PR to its
graph actions. ``sn approve`` tags the merge commit with a machine-readable
contract block (``graph-merged: …``) plus a grounded human summary. The
contract counts docs staged for ordinary quorum review separately from
accepted names, so a successful fold-back never implies those docs are ready
for full export.

These tests pin that contract against a LOCAL bare repo — a real merge commit,
``gh`` and the notes model mocked, no live GitHub and no live LLM:

* the deterministic contract block comes first and is what the idempotency
  guard parses;
* the tag is created on the merge commit and pushed to the target remote;
* a second fold-back is refused (the contract tag already exists);
* ``--undo`` removes the tag (local + remote);
* the grounded summary is appended when synthesis succeeds, and a synthesis
  failure writes the deterministic block alone — never blocking the fold-back.
"""

from __future__ import annotations

import subprocess
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest

from imas_codex.standard_names.promote import (
    CONTRACT_MARKER,
    ApprovalChange,
    ApprovalReport,
    FoldBackTagReport,
    approval_tag_name,
    build_contract_block,
    build_merge_tag_message,
    create_fold_back_tag,
    delete_fold_back_tag,
    fetch_pr_evidence,
    has_contract_tag,
    resolve_tag_remote,
    review_delta_diff,
    run_approval,
    tag_fold_back,
    undo_approval,
)

APPROVAL = "imas_codex.standard_names.promote"


def _git(args: list[str], cwd: Path) -> str:
    return subprocess.run(
        ["git", *args], cwd=str(cwd), capture_output=True, text=True, check=True
    ).stdout


@pytest.fixture()
def merged_repo(tmp_path: Path) -> dict:
    """A checkout whose main carries a merge commit of a review/<rc> branch.

    Returns the checkout dir, the bare origin, the merge-commit SHA, the review
    branch's first-pushed head, and the rc/tag string. The catalog content is
    edited on the branch so the review-delta diff is non-empty.
    """
    bare = tmp_path / "origin.git"
    _git(["init", "--bare", "-b", "main", str(bare)], tmp_path)
    work = tmp_path / "isnc"
    work.mkdir()
    (work / "standard_names").mkdir()
    _git(["init", "-b", "main"], work)
    _git(["config", "user.email", "t@t"], work)
    _git(["config", "user.name", "t"], work)
    _git(["remote", "add", "origin", str(bare)], work)

    yml = work / "standard_names" / "equilibrium.yml"
    yml.write_text("- name: elongation\n  unit: '1'\n  documentation: Original.\n")
    _git(["add", "."], work)
    _git(["commit", "-q", "-m", "base catalog"], work)
    _git(["push", "-q", "origin", "main"], work)

    rc = "v0.1.0rc1"
    _git(["checkout", "-q", "-b", f"review/{rc}"], work)
    yml.write_text("- name: elongation\n  unit: '1'\n  documentation: Batch add.\n")
    _git(["add", "."], work)
    _git(["commit", "-q", "-m", "review batch"], work)
    branch_first = _git(["rev-parse", "HEAD"], work).strip()

    _git(["checkout", "-q", "main"], work)
    _git(["merge", "-q", "--no-ff", f"review/{rc}", "-m", "Merge review batch"], work)
    merge_commit = _git(["rev-parse", "HEAD"], work).strip()
    _git(["push", "-q", "origin", "main"], work)

    return {
        "work": work,
        "bare": bare,
        "merge_commit": merge_commit,
        "branch_first": branch_first,
        "rc": rc,
    }


@pytest.fixture()
def additive_repo(tmp_path: Path) -> dict:
    """A bare fork whose main starts blank and receives one two-entry batch."""
    bare = tmp_path / "origin.git"
    _git(["init", "--bare", "-b", "main", str(bare)], tmp_path)
    work = tmp_path / "isnc"
    work.mkdir()
    _git(["init", "-b", "main"], work)
    _git(["config", "user.email", "t@t"], work)
    _git(["config", "user.name", "t"], work)
    _git(["remote", "add", "origin", str(bare)], work)
    (work / "README.md").write_text("Catalog.\n", encoding="utf-8")
    _git(["add", "README.md"], work)
    _git(["commit", "-q", "-m", "blank catalog"], work)
    _git(["push", "-q", "origin", "main"], work)

    (work / "standard_names").mkdir()
    catalog = work / "standard_names" / "equilibrium.yml"
    merged_bytes = (
        "- name: approved_name\n"
        "  unit: A\n"
        "  documentation: Approved bytes.\n"
        "- name: refused_name\n"
        "  unit: V\n"
        "  documentation: Refused bytes must not survive.\n"
    )
    catalog.write_text(merged_bytes, encoding="utf-8")
    _git(["checkout", "-q", "-b", "review/batch"], work)
    _git(["add", "standard_names/equilibrium.yml"], work)
    _git(["commit", "-q", "-m", "review batch"], work)
    _git(["checkout", "-q", "main"], work)
    _git(["merge", "-q", "--no-ff", "review/batch", "-m", "Merge review batch"], work)
    merge_commit = _git(["rev-parse", "HEAD"], work).strip()
    _git(["push", "-q", "origin", "main"], work)
    return {
        "work": work,
        "bare": bare,
        "merge_commit": merge_commit,
        "base_ref": f"{merge_commit}^1",
        "catalog": catalog,
        "merged_bytes": merged_bytes,
    }


def _report(accepted=1, staged=0, auto=3, contested=0) -> ApprovalReport:
    r = ApprovalReport()
    r.accepted = [f"n{i}" for i in range(accepted)]
    r.staged_for_review = [f"s{i}" for i in range(staged)]
    r.auto_approved = [f"a{i}" for i in range(auto)]
    r.contested = [{"sn_id": f"c{i}"} for i in range(contested)]
    return r


def _fold_additive_batch(repo: dict, *, contest_refused: bool) -> ApprovalReport:
    changes = (
        [
            ApprovalChange(
                sn_id="refused_name",
                axis="docs",
                new_value="Refused bytes must not survive.",
            )
        ]
        if contest_refused
        else []
    )
    edit_plan = MagicMock(blocked=None, successor=None, run_id="approval-review")

    def approve(name: str, **_kwargs) -> bool:
        return name == "approved_name" or not contest_refused

    with (
        patch(f"{APPROVAL}.read_pr_changes", return_value=changes),
        patch(f"{APPROVAL}._name_exists", return_value=True),
        patch(f"{APPROVAL}.apply_edit", return_value=edit_plan),
        patch(f"{APPROVAL}._score_proposal", return_value=0.2),
        patch(f"{APPROVAL}._contest"),
        patch(f"{APPROVAL}.mark_catalog_name_approved", side_effect=approve),
    ):
        return run_approval(
            isnc_dir=repo["work"],
            base_ref=repo["base_ref"],
            threshold=0.85,
            catalog_pr_number=7,
            catalog_pr_url="https://github.com/fork/catalog/pull/7",
            catalog_merge_commit_sha=repo["merge_commit"],
            batch=["approved_name", "refused_name"],
            gc=MagicMock(),
        )


class TestApprovedCatalogMaterialization:
    def test_clean_fold_back_leaves_main_equal_to_the_approved_set(self, additive_repo):
        report = _fold_additive_batch(additive_repo, contest_refused=False)

        assert report.auto_approved == ["approved_name", "refused_name"]
        assert additive_repo["catalog"].read_text() == additive_repo["merged_bytes"]
        assert (
            _git(
                ["show", "origin/main:standard_names/equilibrium.yml"],
                additive_repo["work"],
            )
            == additive_repo["merged_bytes"]
        )
        assert (
            _git(["rev-parse", "origin/main"], additive_repo["work"]).strip()
            == (additive_repo["merge_commit"])
        )

    def test_contested_entry_is_removed_from_main_by_a_correction_commit(
        self, additive_repo
    ):
        report = _fold_additive_batch(additive_repo, contest_refused=True)

        assert report.auto_approved == ["approved_name"]
        assert report.contested == [
            {"sn_id": "refused_name", "target_id": "refused_name", "score": 0.2}
        ]
        expected = (
            "- name: approved_name\n  unit: A\n  documentation: Approved bytes.\n"
        )
        assert additive_repo["catalog"].read_text() == expected
        assert "Refused bytes must not survive" not in _git(
            ["show", "origin/main:standard_names/equilibrium.yml"],
            additive_repo["work"],
        )
        assert (
            _git(["rev-parse", "origin/main"], additive_repo["work"]).strip()
            != (additive_repo["merge_commit"])
        )

    def test_undo_restores_catalog_main_and_graph_lifecycle(self, additive_repo):
        report = _fold_additive_batch(additive_repo, contest_refused=True)
        tag = "approval-receipt"
        ok, err = create_fold_back_tag(
            additive_repo["work"],
            tag=tag,
            merge_commit=additive_repo["merge_commit"],
            message=build_contract_block(
                pr_number=7,
                pr_url="https://github.com/fork/catalog/pull/7",
                batch_artifact="batch.sn_names.yaml",
                report=report,
                timestamp="T",
            ),
            remote="origin",
        )
        assert ok, err

        gc = MagicMock()
        gc.query.side_effect = [
            [{"id": "approved_name"}],
            [{"id": "refused_name"}],
        ]
        graph = undo_approval(
            pr_number=7,
            batch=["approved_name", "refused_name"],
            gc=gc,
        )
        ok, err = delete_fold_back_tag(additive_repo["work"], tag=tag, remote="origin")

        assert ok, err
        assert graph.demoted == ["approved_name"]
        assert graph.contested_reverted == ["refused_name"]
        assert additive_repo["catalog"].read_text() == additive_repo["merged_bytes"]
        assert (
            _git(
                ["show", "origin/main:standard_names/equilibrium.yml"],
                additive_repo["work"],
            )
            == additive_repo["merged_bytes"]
        )
        assert not has_contract_tag(additive_repo["work"], tag)


# ---------------------------------------------------------------------------
# Pure message assembly
# ---------------------------------------------------------------------------


class TestContractMessage:
    def test_contract_block_first_line_is_the_marker(self):
        block = build_contract_block(
            pr_number=7,
            pr_url="https://github.com/o/r/pull/7",
            batch_artifact="v0.1.0rc1.sn_names.yaml",
            report=_report(accepted=1, staged=2, auto=3, contested=1),
            timestamp="2026-07-22T00:00:00+00:00",
        )
        lines = block.splitlines()
        assert lines[0] == f"{CONTRACT_MARKER} 2026-07-22T00:00:00+00:00"
        assert "#7" in block and "pull/7" in block
        assert "v0.1.0rc1.sn_names.yaml" in block
        assert "approved=1" in block
        assert "staged_for_review=2" in block
        assert "auto_approved=3" in block
        assert "contested=1" in block

    def test_message_with_notes_puts_contract_first_then_prose(self):
        block = build_contract_block(
            pr_number=7,
            pr_url="u",
            batch_artifact="b",
            report=_report(),
            timestamp="T",
        )
        msg = build_merge_tag_message(block, "Reviewers renamed one entry.")
        # Contract still parses from line 1; prose sits below a separator.
        assert msg.startswith(f"{CONTRACT_MARKER} T")
        assert "Reviewers renamed one entry." in msg
        assert "\n---\n" in msg
        # The idempotency contract is the first line regardless of the prose.
        assert msg.splitlines()[0].startswith(CONTRACT_MARKER)

    def test_message_without_notes_is_contract_only(self):
        block = build_contract_block(
            pr_number=1,
            pr_url="u",
            batch_artifact=None,
            report=_report(),
            timestamp="T",
        )
        assert build_merge_tag_message(block, "") == block
        assert build_merge_tag_message(block, "   ") == block


class TestApprovalTagName:
    def test_review_branch_yields_rc(self):
        assert approval_tag_name("review/v0.2.0rc65") == "v0.2.0rc65"

    def test_release_branch_yields_version(self):
        assert approval_tag_name("release/v1.0.0") == "v1.0.0"

    def test_other_branch_yields_none(self):
        assert approval_tag_name("main") is None
        assert approval_tag_name("") is None


# ---------------------------------------------------------------------------
# Tag lifecycle against a real bare repo
# ---------------------------------------------------------------------------


class TestTagLifecycle:
    def test_create_tags_merge_commit_and_pushes(self, merged_repo):
        work, merge_commit, rc = (
            merged_repo["work"],
            merged_repo["merge_commit"],
            merged_repo["rc"],
        )
        block = build_contract_block(
            pr_number=7, pr_url="u", batch_artifact="b", report=_report(), timestamp="T"
        )
        ok, err = create_fold_back_tag(
            work,
            tag=rc,
            merge_commit=merge_commit,
            message=build_merge_tag_message(block, "notes"),
            remote="origin",
        )
        assert ok and err is None
        # Tag exists locally, points at the merge commit, carries the contract.
        assert has_contract_tag(work, rc)
        assert _git(["rev-list", "-n1", rc], work).strip() == merge_commit
        # Pushed to the bare origin.
        assert rc in _git(["ls-remote", "--tags", "origin"], work)

    def test_second_fold_back_is_refused_via_contract_tag(self, merged_repo):
        work, merge_commit, rc = (
            merged_repo["work"],
            merged_repo["merge_commit"],
            merged_repo["rc"],
        )
        assert not has_contract_tag(work, rc)
        block = build_contract_block(
            pr_number=7, pr_url="u", batch_artifact="b", report=_report(), timestamp="T"
        )
        create_fold_back_tag(
            work, tag=rc, merge_commit=merge_commit, message=block, remote="origin"
        )
        # The idempotency guard sees the contract tag and would refuse.
        assert has_contract_tag(work, rc)

    def test_delete_removes_local_and_remote(self, merged_repo):
        work, merge_commit, rc = (
            merged_repo["work"],
            merged_repo["merge_commit"],
            merged_repo["rc"],
        )
        block = build_contract_block(
            pr_number=7, pr_url="u", batch_artifact="b", report=_report(), timestamp="T"
        )
        create_fold_back_tag(
            work, tag=rc, merge_commit=merge_commit, message=block, remote="origin"
        )
        assert rc in _git(["ls-remote", "--tags", "origin"], work)

        ok, err = delete_fold_back_tag(work, tag=rc, remote="origin")
        assert ok, err
        assert not has_contract_tag(work, rc)
        assert rc not in _git(["ls-remote", "--tags", "origin"], work)

    def test_push_failure_is_reported_and_rolls_back_local(self, merged_repo):
        work, merge_commit, rc = (
            merged_repo["work"],
            merged_repo["merge_commit"],
            merged_repo["rc"],
        )
        block = build_contract_block(
            pr_number=7, pr_url="u", batch_artifact="b", report=_report(), timestamp="T"
        )
        ok, err = create_fold_back_tag(
            work, tag=rc, merge_commit=merge_commit, message=block, remote="nosuch"
        )
        assert not ok
        assert err and "nosuch" in err
        # No dangling local tag left behind after the failed push.
        assert not has_contract_tag(work, rc)


# ---------------------------------------------------------------------------
# Review-delta diff (what reviewers changed) — real git, no graph
# ---------------------------------------------------------------------------


class TestReviewDelta:
    def test_diff_is_scoped_to_standard_names(self, merged_repo):
        work, merge_commit = merged_repo["work"], merged_repo["merge_commit"]
        # branch_first == merge tip content here, so diff first→merge is empty;
        # diff base-main→merge shows the batch add. Use the PR base (^1).
        base = _git(["rev-parse", f"{merge_commit}^1"], work).strip()
        delta = review_delta_diff(work, base_oid=base, merge_commit=merge_commit)
        assert "standard_names/equilibrium.yml" in delta
        assert "Batch add." in delta

    def test_missing_oids_return_empty(self, merged_repo):
        work = merged_repo["work"]
        assert review_delta_diff(work, base_oid=None, merge_commit="x") == ""
        assert review_delta_diff(work, base_oid="x", merge_commit=None) == ""


# ---------------------------------------------------------------------------
# gh evidence gathering (subprocess mocked)
# ---------------------------------------------------------------------------


class TestFetchPrEvidence:
    def _gh(self, payload, rc=0, err=""):
        import json as _json
        from types import SimpleNamespace

        return SimpleNamespace(returncode=rc, stdout=_json.dumps(payload), stderr=err)

    def test_parses_body_comments_reviews_commits(self):
        payload = {
            "body": "PR description",
            "comments": [{"author": {"login": "rev"}, "body": "looks good"}],
            "reviews": [
                {"author": {"login": "rev"}, "body": "approve", "state": "APPROVED"}
            ],
            "commits": [{"oid": "c1", "messageHeadline": "publish batch"}],
        }
        with patch(f"{APPROVAL}.subprocess.run", return_value=self._gh(payload)):
            evidence = fetch_pr_evidence("https://github.com/o/r/pull/7")
        assert evidence["body"] == "PR description"
        assert evidence["commits"][0]["oid"] == "c1"

    def test_gh_failure_returns_empty_never_raises(self):
        with patch(
            f"{APPROVAL}.subprocess.run", return_value=self._gh({}, rc=1, err="no auth")
        ):
            assert fetch_pr_evidence("https://github.com/o/r/pull/7") == {}


class TestResolveTagRemote:
    def test_matches_pr_url_owner_repo_to_a_remote(self, tmp_path):
        work = tmp_path / "isnc"
        work.mkdir()
        _git(["init", "-b", "main"], work)
        _git(["remote", "add", "origin", "git@github.com:fork/cat.git"], work)
        _git(["remote", "add", "upstream", "git@github.com:org/cat.git"], work)
        assert (
            resolve_tag_remote(work, "https://github.com/org/cat/pull/3") == "upstream"
        )
        assert (
            resolve_tag_remote(work, "https://github.com/fork/cat/pull/3") == "origin"
        )

    def test_unmatched_url_defaults_to_origin(self, tmp_path):
        work = tmp_path / "isnc"
        work.mkdir()
        _git(["init", "-b", "main"], work)
        assert resolve_tag_remote(work, "not-a-github-url") == "origin"


# ---------------------------------------------------------------------------
# tag_fold_back orchestrator — evidence + notes injected, real tag write
# ---------------------------------------------------------------------------


class TestTagFoldBack:
    def _base_kwargs(self, merged_repo):
        return {
            "isnc_dir": merged_repo["work"],
            "head_ref": f"review/{merged_repo['rc']}",
            "merge_commit": merged_repo["merge_commit"],
            "pr_number": 7,
            "pr_url": "https://github.com/o/r/pull/7",
            "batch_artifact": "v0.1.0rc1.sn_names.yaml",
            "report": _report(accepted=2, auto=5, contested=1),
            "remote": "origin",
            "pr_evidence": {"body": "d", "comments": [], "reviews": [], "commits": []},
        }

    def test_happy_path_creates_tag_with_grounded_notes(self, merged_repo):
        seen: dict = {}

        def notes_builder(**kw):
            seen.update(kw)
            return "Reviewers accepted the batch; one docs edit re-reviewed."

        out = tag_fold_back(
            **self._base_kwargs(merged_repo),
            notes_builder=notes_builder,
        )
        assert isinstance(out, FoldBackTagReport)
        assert out.created and out.pushed and out.error is None
        assert out.notes_included is True
        assert out.tag == merged_repo["rc"]
        # The builder received the grounded evidence fields.
        assert set(seen) >= {
            "pr_description",
            "conversation",
            "commit_messages",
            "review_delta",
        }
        # The written tag carries the contract block AND the prose summary.
        msg = _git(
            ["tag", "-l", merged_repo["rc"], "--format=%(contents)"],
            merged_repo["work"],
        )
        assert msg.startswith(CONTRACT_MARKER)
        assert "one docs edit re-reviewed" in msg

    def test_notes_failure_writes_deterministic_block_only(self, merged_repo):
        def boom(**kw):
            raise RuntimeError("model down")

        out = tag_fold_back(
            **self._base_kwargs(merged_repo),
            notes_builder=boom,
        )
        # The fold-back is never blocked by a notes failure.
        assert out.created and out.pushed
        assert out.notes_included is False
        msg = _git(
            ["tag", "-l", merged_repo["rc"], "--format=%(contents)"],
            merged_repo["work"],
        )
        assert msg.startswith(CONTRACT_MARKER)
        assert "---" not in msg  # no prose separator

    def test_no_notes_skips_the_builder(self, merged_repo):
        called = {"n": 0}

        def notes_builder(**kw):
            called["n"] += 1
            return "should not appear"

        out = tag_fold_back(
            **self._base_kwargs(merged_repo),
            include_notes=False,
            notes_builder=notes_builder,
        )
        assert out.created and out.notes_included is False
        assert called["n"] == 0

    def test_underivable_tag_reports_error_without_writing(self, merged_repo):
        kwargs = self._base_kwargs(merged_repo)
        kwargs["head_ref"] = "main"
        out = tag_fold_back(**kwargs, notes_builder=lambda **k: "x")
        assert not out.created
        assert out.error and "version tag" in out.error


# ---------------------------------------------------------------------------
# CLI flow — sn approve writes / refuses / removes the receipt tag
# ---------------------------------------------------------------------------


class TestApprovalCliTag:
    def _resolved(self, merged_repo):
        from imas_codex.standard_names.promote import ResolvedPr

        return ResolvedPr(
            number=7,
            url="https://github.com/o/r/pull/7",
            merge_commit=merged_repo["merge_commit"],
            head_ref=f"review/{merged_repo['rc']}",
            base_ref="main",
        )

    def test_approval_creates_the_contract_tag(self, merged_repo):
        from click.testing import CliRunner

        from imas_codex.cli.sn import sn

        work, rc = merged_repo["work"], merged_repo["rc"]
        with (
            patch(
                f"{APPROVAL}.resolve_merged_pr",
                return_value=self._resolved(merged_repo),
            ),
            patch(f"{APPROVAL}.run_approval", return_value=_report(accepted=2, auto=5)),
        ):
            result = CliRunner().invoke(
                sn,
                [
                    "approve",
                    "--isnc",
                    str(work),
                    "--pr",
                    "https://github.com/o/r/pull/7",
                    "--no-notes",
                ],
            )
        assert result.exit_code == 0, result.output
        assert has_contract_tag(work, rc)

    def test_existing_rc_tag_does_not_block_approval(self, merged_repo):
        from click.testing import CliRunner

        from imas_codex.cli.sn import sn

        work, rc = merged_repo["work"], merged_repo["rc"]
        _git(["tag", "-a", rc, merged_repo["branch_first"], "-m", "RC cut"], work)
        _git(["push", "origin", f"refs/tags/{rc}"], work)
        assert not has_contract_tag(work, rc)

        with (
            patch(
                f"{APPROVAL}.resolve_merged_pr",
                return_value=self._resolved(merged_repo),
            ),
            patch(f"{APPROVAL}.run_approval", return_value=_report(accepted=2, auto=5)),
        ):
            result = CliRunner().invoke(
                sn,
                [
                    "approve",
                    "--isnc",
                    str(work),
                    "--pr",
                    "https://github.com/o/r/pull/7",
                    "--no-notes",
                ],
            )

        assert result.exit_code == 0, result.output
        assert has_contract_tag(work, rc)
        assert (
            _git(["rev-list", "-n1", rc], work).strip() == merged_repo["merge_commit"]
        )

    def test_approval_refuses_when_already_folded_back(self, merged_repo):
        from click.testing import CliRunner

        from imas_codex.cli.sn import sn

        work, rc, merge_commit = (
            merged_repo["work"],
            merged_repo["rc"],
            merged_repo["merge_commit"],
        )
        block = build_contract_block(
            pr_number=7, pr_url="u", batch_artifact="b", report=_report(), timestamp="T"
        )
        create_fold_back_tag(
            work, tag=rc, merge_commit=merge_commit, message=block, remote="origin"
        )
        with (
            patch(
                f"{APPROVAL}.resolve_merged_pr",
                return_value=self._resolved(merged_repo),
            ),
            patch(f"{APPROVAL}.run_approval") as m_run,
        ):
            result = CliRunner().invoke(
                sn,
                [
                    "approve",
                    "--isnc",
                    str(work),
                    "--pr",
                    "https://github.com/o/r/pull/7",
                    "--no-notes",
                ],
            )
        assert result.exit_code != 0
        m_run.assert_not_called()
        assert "already" in result.output.lower() or "folded" in result.output.lower()

    def test_undo_removes_the_contract_tag(self, merged_repo):
        from click.testing import CliRunner

        from imas_codex.cli.sn import sn
        from imas_codex.standard_names.promote import UndoApprovalReport

        work, rc, merge_commit = (
            merged_repo["work"],
            merged_repo["rc"],
            merged_repo["merge_commit"],
        )
        block = build_contract_block(
            pr_number=7, pr_url="u", batch_artifact="b", report=_report(), timestamp="T"
        )
        create_fold_back_tag(
            work, tag=rc, merge_commit=merge_commit, message=block, remote="origin"
        )
        assert has_contract_tag(work, rc)
        with (
            patch(
                f"{APPROVAL}.resolve_merged_pr",
                return_value=self._resolved(merged_repo),
            ),
            patch(
                f"{APPROVAL}.undo_approval",
                return_value=UndoApprovalReport(pr_number=7),
            ),
        ):
            result = CliRunner().invoke(
                sn,
                [
                    "approve",
                    "--isnc",
                    str(work),
                    "--pr",
                    "https://github.com/o/r/pull/7",
                    "--undo",
                ],
            )
        assert result.exit_code == 0, result.output
        assert not has_contract_tag(work, rc)

    def test_undo_restores_the_exact_rc_tag(self, merged_repo):
        from click.testing import CliRunner

        from imas_codex.cli.sn import sn
        from imas_codex.standard_names.promote import UndoApprovalReport

        work, rc = merged_repo["work"], merged_repo["rc"]
        _git(["tag", "-a", rc, merged_repo["branch_first"], "-m", "RC cut"], work)
        original_ref = _git(["rev-parse", f"refs/tags/{rc}"], work).strip()
        _git(["push", "origin", f"refs/tags/{rc}"], work)
        folded = tag_fold_back(
            isnc_dir=work,
            head_ref=f"review/{rc}",
            merge_commit=merged_repo["merge_commit"],
            pr_number=7,
            pr_url="https://github.com/o/r/pull/7",
            batch_artifact=None,
            report=_report(),
            remote="origin",
            include_notes=False,
            timestamp="T",
        )
        assert folded.created and has_contract_tag(work, rc)

        with (
            patch(
                f"{APPROVAL}.resolve_merged_pr",
                return_value=self._resolved(merged_repo),
            ),
            patch(
                f"{APPROVAL}.undo_approval",
                return_value=UndoApprovalReport(pr_number=7),
            ),
        ):
            result = CliRunner().invoke(
                sn,
                [
                    "approve",
                    "--isnc",
                    str(work),
                    "--pr",
                    "https://github.com/o/r/pull/7",
                    "--undo",
                ],
            )

        assert result.exit_code == 0, result.output
        assert not has_contract_tag(work, rc)
        assert _git(["rev-parse", f"refs/tags/{rc}"], work).strip() == original_ref
        assert (
            _git(["rev-list", "-n1", rc], work).strip() == merged_repo["branch_first"]
        )
        assert rc in _git(["ls-remote", "--tags", "origin"], work)


# ---------------------------------------------------------------------------
# A batch RC's version carries semver build metadata (v0.2.0rc65+label)
# ---------------------------------------------------------------------------


@pytest.fixture()
def labelled_merged_repo(tmp_path: Path) -> dict:
    """The merged_repo shape, but the review branch carries a labelled version."""
    bare = tmp_path / "origin.git"
    _git(["init", "--bare", "-b", "main", str(bare)], tmp_path)
    work = tmp_path / "isnc"
    work.mkdir()
    (work / "standard_names").mkdir()
    _git(["init", "-b", "main"], work)
    _git(["config", "user.email", "t@t"], work)
    _git(["config", "user.name", "t"], work)
    _git(["remote", "add", "origin", str(bare)], work)
    yml = work / "standard_names" / "equilibrium.yml"
    yml.write_text("- name: elongation\n  unit: '1'\n  documentation: Original.\n")
    _git(["add", "."], work)
    _git(["commit", "-q", "-m", "base catalog"], work)
    _git(["push", "-q", "origin", "main"], work)

    rc = "v0.1.0rc1+west-task-2e"
    _git(["checkout", "-q", "-b", f"review/{rc}"], work)
    yml.write_text("- name: elongation\n  unit: '1'\n  documentation: Batch add.\n")
    _git(["add", "."], work)
    _git(["commit", "-q", "-m", "review batch"], work)
    _git(["checkout", "-q", "main"], work)
    _git(["merge", "-q", "--no-ff", f"review/{rc}", "-m", "Merge review batch"], work)
    merge_commit = _git(["rev-parse", "HEAD"], work).strip()
    _git(["push", "-q", "origin", "main"], work)
    return {"work": work, "merge_commit": merge_commit, "rc": rc}


class TestLabelledVersionReceipt:
    def test_a_plus_label_is_a_legal_branch_and_tag_ref(self, labelled_merged_repo):
        """'+' is legal in a git ref, so branch and tag both hold the label."""
        work, rc = labelled_merged_repo["work"], labelled_merged_repo["rc"]
        assert f"review/{rc}" in _git(["branch"], work)

    def test_receipt_tag_uses_the_full_labelled_version(self, labelled_merged_repo):
        work, merge_commit, rc = (
            labelled_merged_repo["work"],
            labelled_merged_repo["merge_commit"],
            labelled_merged_repo["rc"],
        )
        out = tag_fold_back(
            isnc_dir=work,
            head_ref=f"review/{rc}",
            merge_commit=merge_commit,
            pr_number=7,
            pr_url="https://github.com/o/r/pull/7",
            batch_artifact=f"{rc}.sn_names.yaml",
            report=_report(),
            remote="origin",
            include_notes=False,
            timestamp="T",
        )
        assert out.error is None
        assert out.tag == rc
        assert out.created and out.pushed
        assert has_contract_tag(work, rc)
        assert _git(["rev-list", "-n1", rc], work).strip() == merge_commit
        assert rc in _git(["ls-remote", "--tags", "origin"], work)

    def test_undo_deletes_the_labelled_receipt_tag(self, labelled_merged_repo):
        from imas_codex.standard_names.promote import delete_fold_back_tag

        work, merge_commit, rc = (
            labelled_merged_repo["work"],
            labelled_merged_repo["merge_commit"],
            labelled_merged_repo["rc"],
        )
        tag_fold_back(
            isnc_dir=work,
            head_ref=f"review/{rc}",
            merge_commit=merge_commit,
            pr_number=7,
            pr_url="u",
            batch_artifact=None,
            report=_report(),
            remote="origin",
            include_notes=False,
            timestamp="T",
        )
        assert has_contract_tag(work, rc)
        ok, err = delete_fold_back_tag(work, tag=rc, remote="origin")
        assert ok, err
        assert not has_contract_tag(work, rc)
        assert rc not in _git(["ls-remote", "--tags", "origin"], work)


class TestApprovalCliLabelledVersion:
    """The CLI must resolve a labelled batch RC end-to-end from the branch name."""

    def _resolved(self, repo):
        from imas_codex.standard_names.promote import ResolvedPr

        return ResolvedPr(
            number=7,
            url="https://github.com/o/r/pull/7",
            merge_commit=repo["merge_commit"],
            head_ref=f"review/{repo['rc']}",
            base_ref="main",
        )

    def test_approval_locates_the_labelled_artifact_and_tags_the_version(
        self, labelled_merged_repo, tmp_path, monkeypatch
    ):
        from click.testing import CliRunner

        from imas_codex.cli.sn import sn

        work, rc = labelled_merged_repo["work"], labelled_merged_repo["rc"]
        # Frozen artifact for this labelled RC, in a temp reviews home.
        reviews = tmp_path / "reviews"
        reviews.mkdir()
        (reviews / f"{rc}.sn_names.yaml").write_text(
            "kind: sn_names\nschema_version: 1\nname: b\nnames:\n  - plasma_current\n",
            encoding="utf-8",
        )
        monkeypatch.setattr(
            "imas_codex.standard_names.catalog_release.default_reviews_dir",
            lambda: reviews,
        )
        seen: dict = {}

        def _run_approval(**kw):
            seen.update(kw)
            return _report(accepted=1, auto=0)

        with (
            patch(
                f"{APPROVAL}.resolve_merged_pr",
                return_value=self._resolved(labelled_merged_repo),
            ),
            patch(f"{APPROVAL}.run_approval", side_effect=_run_approval),
        ):
            result = CliRunner().invoke(
                sn,
                [
                    "approve",
                    "--isnc",
                    str(work),
                    "--pr",
                    "https://github.com/o/r/pull/7",
                    "--no-notes",
                ],
            )
        assert result.exit_code == 0, result.output
        # The artifact was auto-located from the labelled branch name.
        assert seen["batch"] == ["plasma_current"]
        assert f"{rc}.sn_names.yaml" in result.output
        # The receipt tag carries the full labelled version.
        assert has_contract_tag(work, rc)

    def test_second_fold_back_of_a_labelled_rc_is_refused(self, labelled_merged_repo):
        from click.testing import CliRunner

        from imas_codex.cli.sn import sn

        work, rc, merge_commit = (
            labelled_merged_repo["work"],
            labelled_merged_repo["rc"],
            labelled_merged_repo["merge_commit"],
        )
        create_fold_back_tag(
            work,
            tag=rc,
            merge_commit=merge_commit,
            message=build_merge_tag_message(
                build_contract_block(
                    pr_number=7,
                    pr_url="u",
                    batch_artifact=None,
                    report=_report(),
                    timestamp="T",
                )
            ),
            remote="origin",
        )
        with patch(
            f"{APPROVAL}.resolve_merged_pr",
            return_value=self._resolved(labelled_merged_repo),
        ):
            result = CliRunner().invoke(
                sn,
                [
                    "approve",
                    "--isnc",
                    str(work),
                    "--pr",
                    "https://github.com/o/r/pull/7",
                ],
            )
        assert result.exit_code == 1
        assert rc in result.output
