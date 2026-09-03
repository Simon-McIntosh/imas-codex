"""Tests for ``sn approve`` — fold a reviewed catalog PR back into the ledger.

All graph interaction is MOCKED — these tests never touch Neo4j. They pin
the orchestration contract of :func:`run_approval`:

* a human catalog edit is re-attached like ``sn edit`` (candidate + reason,
  ``origin='human'``, ``refine=False``) via :func:`apply_edit`;
* the attached proposal is scored without invoking refine pools;
* a name review score at or above threshold accepts through
  ``persist_reviewed_name``;
* a docs score at or above threshold is not treated as quorum authority: the
  unchanged text is staged for an exact ordinary docs review and remains out
  of full export;
* a score below threshold QUARANTINES + FLAGS (``validation_status`` set to
  ``'quarantined'``) — never accepted, never refined, never mutated;
* a NAME change routes through ``apply_edit``'s rename mode (which carries
  ``PRODUCED_NAME`` provenance through the rename cascade), never
  delete-and-recreate.
"""

from __future__ import annotations

import subprocess
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest

from imas_codex.standard_names.edit import EditPlan
from imas_codex.standard_names.promote import (
    ApprovalChange,
    ApprovalReport,
    read_pr_changes,
    run_approval,
)

APPROVAL = "imas_codex.standard_names.promote"


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _edit_plan(
    *,
    target: str,
    mode: str = "docs",
    successor: str | None = None,
    run_id: str = "sn-edit-20260709T000000Z",
    blocked: str | None = None,
    applied: bool = True,
) -> EditPlan:
    """Build a realistic :class:`EditPlan` for mocking ``apply_edit``."""
    axis = "name" if mode == "rename" else "docs"
    entry = {"rename": "review_name", "docs": "review_docs", "hint": "generate"}[mode]
    return EditPlan(
        target=target,
        mode=mode,
        axis=axis,
        scope="only_self",
        entry=entry,
        successor=successor,
        cascade_planned=[],
        blocked=blocked,
        actions=[],
        applied=applied and blocked is None,
        run_id=None if blocked else run_id,
    )


def _gc_exists(exists: bool = True) -> MagicMock:
    """A GraphClient mock whose existence probe reports (non-)existence."""
    gc = MagicMock(name="gc")
    gc.query.return_value = [{"n": 1 if exists else 0}]
    return gc


# ---------------------------------------------------------------------------
# Accept path
# ---------------------------------------------------------------------------


class TestRunApprovalPassingReviewPath:
    def test_docs_edit_attached_like_sn_edit_with_reason(self):
        change = ApprovalChange(
            sn_id="electron_temperature",
            axis="docs",
            new_value="Electron temperature. Revised by reviewer.",
            old_value="Electron temperature.",
        )
        gc = _gc_exists(True)
        with (
            patch(f"{APPROVAL}.read_pr_changes", return_value=[change]),
            patch(f"{APPROVAL}.apply_edit") as m_apply,
            patch(f"{APPROVAL}._score_proposal", return_value=0.93) as m_score,
            patch(
                f"{APPROVAL}.persist_reviewed_docs", return_value="reviewed"
            ) as m_docs,
            patch(
                f"{APPROVAL}.stage_docs_for_rescore",
                return_value={"ok": True},
            ) as m_stage,
            patch(f"{APPROVAL}.mark_catalog_name_approved") as m_approve,
            patch(f"{APPROVAL}.persist_reviewed_name") as m_name,
        ):
            m_apply.return_value = _edit_plan(target="electron_temperature")
            report = run_approval(
                isnc_dir="/tmp/isnc",
                base_ref="origin/main",
                threshold=0.85,
                catalog_pr_number=12,
                catalog_pr_url="https://example.test/pull/12",
                catalog_merge_commit_sha="abc123",
                gc=gc,
            )

        # Attached exactly like sn edit: the changed field is the candidate +
        # a human reason, origin='human', refine disabled.
        assert m_apply.call_count == 1
        kwargs = m_apply.call_args.kwargs
        assert kwargs["target"] == "electron_temperature"
        assert kwargs["docs"] == change.new_value
        assert kwargs["origin"] == "human"
        assert kwargs["refine"] is False
        assert kwargs["reason"] and "PR" in kwargs["reason"]
        assert "rename" not in kwargs or kwargs.get("rename") is None

        # The aggregate score is recorded fail-closed, then the same text is
        # staged for the complete configured docs quorum.
        m_score.assert_called_once()
        m_docs.assert_called_once()
        m_stage.assert_called_once_with(
            "electron_temperature", run_id="sn-edit-20260709T000000Z"
        )
        m_approve.assert_not_called()
        m_name.assert_not_called()
        assert report.accepted == []
        assert report.staged_for_review == ["electron_temperature"]
        assert report.outcomes[0].decision == "staged_for_review"
        assert report.threshold == 0.85

    def test_docs_score_at_threshold_stages_for_quorum(self):
        change = ApprovalChange(
            sn_id="ion_density", axis="docs", new_value="Ion density."
        )
        gc = _gc_exists(True)
        with (
            patch(f"{APPROVAL}.read_pr_changes", return_value=[change]),
            patch(
                f"{APPROVAL}.apply_edit", return_value=_edit_plan(target="ion_density")
            ),
            patch(f"{APPROVAL}._score_proposal", return_value=0.85),
            patch(
                f"{APPROVAL}.persist_reviewed_docs", return_value="reviewed"
            ) as m_docs,
            patch(
                f"{APPROVAL}.stage_docs_for_rescore",
                return_value={"ok": True},
            ),
        ):
            report = run_approval(isnc_dir="/x", base_ref="b", threshold=0.85, gc=gc)
        m_docs.assert_called_once()
        assert report.accepted == []
        assert report.staged_for_review == ["ion_density"]
        assert not report.quarantined


# ---------------------------------------------------------------------------
# Quarantine path (score below threshold)
# ---------------------------------------------------------------------------


class TestRunApprovalContestPath:
    def test_low_score_contests_not_accepted_not_refined(self):
        change = ApprovalChange(sn_id="ion_density", axis="docs", new_value="Bad docs.")
        gc = _gc_exists(True)
        with (
            patch(f"{APPROVAL}.read_pr_changes", return_value=[change]),
            patch(
                f"{APPROVAL}.apply_edit", return_value=_edit_plan(target="ion_density")
            ),
            patch(f"{APPROVAL}._score_proposal", return_value=0.50),
            patch(f"{APPROVAL}.persist_reviewed_docs") as m_docs,
            patch(f"{APPROVAL}.persist_reviewed_name") as m_name,
        ):
            report = run_approval(isnc_dir="/x", base_ref="b", threshold=0.85, gc=gc)

        # NOT accepted — no promote path taken.
        m_docs.assert_not_called()
        m_name.assert_not_called()
        assert "ion_density" not in report.accepted

        # A failed reviewer edit is contested, not quarantined.
        assert any(c["sn_id"] == "ion_density" for c in report.contested)
        assert not report.quarantined
        # The contested signal (name_stage='contested') is written to the graph.
        contest_queries = [
            c.args[0]
            for c in gc.query.call_args_list
            if c.args and "contested" in c.args[0]
        ]
        assert contest_queries, "expected a gc.query setting name_stage='contested'"
        assert "name_stage = 'contested'" in contest_queries[0]

    def test_full_review_runs_but_refine_never_invoked(self):
        """FULL review is exercised; NO refine pool is ever entered."""
        changes = [
            ApprovalChange(sn_id="a_name", axis="docs", new_value="Doc A."),
            ApprovalChange(sn_id="b_name", axis="docs", new_value="Doc B."),
        ]
        gc = _gc_exists(True)
        with (
            patch(f"{APPROVAL}.read_pr_changes", return_value=changes),
            patch(
                f"{APPROVAL}.apply_edit",
                side_effect=lambda **k: _edit_plan(target=k["target"]),
            ),
            patch(f"{APPROVAL}._score_proposal", side_effect=[0.9, 0.4]) as m_score,
            patch(f"{APPROVAL}.persist_reviewed_docs", return_value="reviewed"),
            patch(f"{APPROVAL}.stage_docs_for_rescore", return_value={"ok": True}),
            patch(
                "imas_codex.standard_names.graph_ops.claim_refine_name_batch"
            ) as m_rn,
            patch(
                "imas_codex.standard_names.graph_ops.claim_refine_docs_batch"
            ) as m_rd,
            patch(
                "imas_codex.standard_names.workers.process_refine_name_batch"
            ) as m_prn,
            patch(
                "imas_codex.standard_names.workers.process_refine_docs_batch"
            ) as m_prd,
        ):
            report = run_approval(isnc_dir="/x", base_ref="b", threshold=0.85, gc=gc)

        # Review scored every matched proposal…
        assert m_score.call_count == 2
        # …but NOT ONE refine claim/process was invoked.
        m_rn.assert_not_called()
        m_rd.assert_not_called()
        m_prn.assert_not_called()
        m_prd.assert_not_called()
        assert report.accepted == []
        assert report.staged_for_review == ["a_name"]
        assert any(c["sn_id"] == "b_name" for c in report.contested)


# ---------------------------------------------------------------------------
# Name edit → rename cascade
# ---------------------------------------------------------------------------


class TestRunApprovalNameEdit:
    def test_name_edit_routes_through_rename_mode(self):
        change = ApprovalChange(
            sn_id="elongation",
            axis="name",
            new_value="elongation_of_closed_flux_surface",
        )
        gc = _gc_exists(True)
        with (
            patch(f"{APPROVAL}.read_pr_changes", return_value=[change]),
            patch(f"{APPROVAL}.apply_edit") as m_apply,
            patch(f"{APPROVAL}._score_proposal", return_value=0.95) as m_score,
            patch(
                f"{APPROVAL}.persist_reviewed_name", return_value="accepted"
            ) as m_name,
            patch(f"{APPROVAL}.persist_reviewed_docs") as m_docs,
        ):
            m_apply.return_value = _edit_plan(
                target="elongation",
                mode="rename",
                successor="elongation_of_closed_flux_surface",
            )
            report = run_approval(isnc_dir="/x", base_ref="b", threshold=0.85, gc=gc)

        # A NAME change rides rename mode (cascade-carrying), never docs/delete.
        kwargs = m_apply.call_args.kwargs
        assert kwargs["rename"] == "elongation_of_closed_flux_surface"
        assert kwargs.get("docs") is None
        assert kwargs["refine"] is False
        # The scored + accepted target is the rename SUCCESSOR, not the old id.
        assert m_score.call_args.args[0] == "elongation_of_closed_flux_surface"
        m_name.assert_called_once()
        m_docs.assert_not_called()
        assert "elongation_of_closed_flux_surface" in report.accepted


# ---------------------------------------------------------------------------
# Unmatched / blocked / dry-run
# ---------------------------------------------------------------------------


class TestRunApprovalEdgeCases:
    def test_unmatched_id_is_reported_without_attaching(self):
        change = ApprovalChange(sn_id="ghost_name", axis="docs", new_value="x")
        gc = _gc_exists(False)
        with (
            patch(f"{APPROVAL}.read_pr_changes", return_value=[change]),
            patch(f"{APPROVAL}.apply_edit") as m_apply,
            patch(f"{APPROVAL}._score_proposal") as m_score,
        ):
            report = run_approval(isnc_dir="/x", base_ref="b", gc=gc)
        m_apply.assert_not_called()
        m_score.assert_not_called()
        assert "ghost_name" in report.unmatched

    def test_blocked_edit_is_recorded_and_not_scored(self):
        change = ApprovalChange(sn_id="frozen_name", axis="docs", new_value="x")
        gc = _gc_exists(True)
        with (
            patch(f"{APPROVAL}.read_pr_changes", return_value=[change]),
            patch(
                f"{APPROVAL}.apply_edit",
                return_value=_edit_plan(
                    target="frozen_name", blocked="target ineligible stage"
                ),
            ),
            patch(f"{APPROVAL}._score_proposal") as m_score,
            patch(f"{APPROVAL}.persist_reviewed_docs") as m_docs,
        ):
            report = run_approval(isnc_dir="/x", base_ref="b", gc=gc)
        m_score.assert_not_called()
        m_docs.assert_not_called()
        assert any(b["sn_id"] == "frozen_name" for b in report.blocked)

    def test_dry_run_attaches_nothing(self):
        change = ApprovalChange(sn_id="ion_density", axis="docs", new_value="x")
        gc = _gc_exists(True)
        with (
            patch(f"{APPROVAL}.read_pr_changes", return_value=[change]),
            patch(f"{APPROVAL}.apply_edit") as m_apply,
            patch(f"{APPROVAL}._score_proposal") as m_score,
            patch(f"{APPROVAL}.persist_reviewed_docs") as m_docs,
        ):
            report = run_approval(isnc_dir="/x", base_ref="b", dry_run=True, gc=gc)
        m_apply.assert_not_called()
        m_score.assert_not_called()
        m_docs.assert_not_called()
        assert report.dry_run is True
        assert isinstance(report, ApprovalReport)


# ---------------------------------------------------------------------------
# read_pr_changes — real git worktree, no graph
# ---------------------------------------------------------------------------


def _git(args: list[str], cwd: Path) -> str:
    return subprocess.run(
        ["git", *args], cwd=str(cwd), capture_output=True, text=True, check=True
    ).stdout


@pytest.fixture()
def catalog_repo(tmp_path: Path) -> tuple[Path, str]:
    """A tiny git catalog repo; returns (dir, base_sha) with one committed entry."""
    repo = tmp_path / "isnc"
    (repo / "standard_names").mkdir(parents=True)
    _git(["init", "-q"], repo)
    _git(["config", "user.email", "t@t"], repo)
    _git(["config", "user.name", "t"], repo)
    yml = repo / "standard_names" / "equilibrium.yml"
    yml.write_text(
        "- name: elongation\n"
        "  kind: scalar\n"
        "  unit: '1'\n"
        "  documentation: Original elongation docs.\n"
    )
    _git(["add", "."], repo)
    _git(["commit", "-q", "-m", "base"], repo)
    base = _git(["rev-parse", "HEAD"], repo).strip()
    return repo, base


class TestReadPrChanges:
    def test_docs_change_detected(self, catalog_repo):
        repo, base = catalog_repo
        yml = repo / "standard_names" / "equilibrium.yml"
        yml.write_text(
            "- name: elongation\n"
            "  kind: scalar\n"
            "  unit: '1'\n"
            "  documentation: Revised elongation docs by the reviewer.\n"
        )
        changes = read_pr_changes(repo, base)
        docs = [c for c in changes if c.axis == "docs"]
        assert len(docs) == 1
        assert docs[0].sn_id == "elongation"
        assert "Revised" in docs[0].new_value

    def test_no_change_returns_empty(self, catalog_repo):
        repo, base = catalog_repo
        assert read_pr_changes(repo, base) == []

    def test_name_rename_paired_by_unit_and_kind(self, catalog_repo):
        repo, base = catalog_repo
        yml = repo / "standard_names" / "equilibrium.yml"
        yml.write_text(
            "- name: elongation_of_closed_flux_surface\n"
            "  kind: scalar\n"
            "  unit: '1'\n"
            "  documentation: Original elongation docs.\n"
        )
        changes = read_pr_changes(repo, base)
        names = [c for c in changes if c.axis == "name"]
        assert len(names) == 1
        assert names[0].sn_id == "elongation"
        assert names[0].new_value == "elongation_of_closed_flux_surface"


# ---------------------------------------------------------------------------
# PR-URL resolution (GitHub REST transport mocked)
# ---------------------------------------------------------------------------


def _rest_pull(**overrides):
    """A REST pull-request payload with the fields the resolver reads."""
    payload = {
        "number": 7,
        "html_url": "https://github.com/o/r/pull/7",
        "state": "closed",
        "merged": True,
        "merge_commit_sha": "cafe1234",
        "user": {"login": "reviewer"},
        "head": {"ref": "review/v0.2.0rc65"},
        "base": {"ref": "main"},
    }
    payload.update(overrides)
    return payload


def _patch_rest(status=200, payload=None, recorder=None):
    """Bind the shared transport to a canned response, recording every path."""

    def fake_call(method, path, *, payload=None, token=None):
        if recorder is not None:
            recorder.append((method, path))
        return fake_call.status, fake_call.body

    fake_call.status = status
    fake_call.body = payload
    return patch("imas_codex.graph.ghcr.github_api_call", fake_call)


class TestResolveMergedPr:
    def test_resolves_merged_pr(self):
        from imas_codex.standard_names.promote import resolve_merged_pr

        seen: list[tuple[str, str]] = []
        with _patch_rest(payload=_rest_pull(), recorder=seen):
            r = resolve_merged_pr("https://github.com/o/r/pull/7")
        assert r.number == 7
        assert r.merge_commit == "cafe1234"
        assert r.head_ref == "review/v0.2.0rc65"
        assert r.reviewer_actor == "reviewer"
        assert seen == [("GET", "/repos/o/r/pulls/7")]

    def test_merged_state_is_read_off_the_merge_record_not_the_state_field(self):
        """REST reports a merged pull request as ``closed``."""
        from imas_codex.standard_names.promote import resolve_merged_pr

        payload = _rest_pull(merged=False, merged_at="2026-09-01T00:00:00Z")
        with _patch_rest(payload=payload):
            assert resolve_merged_pr("https://github.com/o/r/pull/7").number == 7

    def test_unmerged_pr_rejected(self):
        from imas_codex.standard_names.promote import resolve_merged_pr

        payload = _rest_pull(
            state="open", merged=False, merged_at=None, merge_commit_sha=None
        )
        with (
            _patch_rest(payload=payload),
            pytest.raises(ValueError, match="not merged"),
        ):
            resolve_merged_pr("https://github.com/o/r/pull/7")

    def test_rejected_read_surfaces_what_github_said(self):
        from imas_codex.standard_names.promote import resolve_merged_pr

        with (
            _patch_rest(status=401, payload={"message": "Bad credentials"}),
            pytest.raises(ValueError, match="Bad credentials"),
        ):
            resolve_merged_pr("https://github.com/o/r/pull/7")

    def test_url_that_is_not_a_pull_request_is_refused(self):
        from imas_codex.standard_names.promote import resolve_merged_pr

        with pytest.raises(ValueError, match="pull-request URL"):
            resolve_merged_pr("https://github.com/o/r/issues/7")

    def test_no_pull_request_call_shells_out(self):
        from imas_codex.standard_names.promote import resolve_merged_pr

        def fail(*args, **kwargs):
            raise AssertionError(f"pull-request work shelled out to {args[0]!r}")

        with (
            _patch_rest(payload=_rest_pull()),
            patch("imas_codex.standard_names.promote.subprocess.run", fail),
        ):
            assert resolve_merged_pr("https://github.com/o/r/pull/7").number == 7


# ---------------------------------------------------------------------------
# Batch RCs carry the batch label as semver build metadata (v0.2.0rc65+label)
# ---------------------------------------------------------------------------


class TestBatchLabelledVersionRoundTrip:
    """A release cut against a batch must survive release → branch → merge.

    The version string carrying a ``+label`` suffix is the single key that
    names the tag, the review branch, and the frozen artifact — so the merge
    side must parse it back out of the branch name unchanged.
    """

    def test_approval_tag_name_keeps_the_label(self):
        from imas_codex.standard_names.promote import approval_tag_name

        assert (
            approval_tag_name("review/v0.2.0rc65+west-task-2e")
            == "v0.2.0rc65+west-task-2e"
        )

    def test_plain_and_release_branches_are_unaffected(self):
        from imas_codex.standard_names.promote import approval_tag_name

        assert approval_tag_name("review/v0.2.0rc65") == "v0.2.0rc65"
        assert approval_tag_name("release/v1.0.0") == "v1.0.0"

    def test_resolve_merged_pr_carries_a_labelled_head_ref(self):
        from imas_codex.standard_names.promote import resolve_merged_pr

        payload = _rest_pull(
            number=9,
            html_url="https://github.com/o/r/pull/9",
            merge_commit_sha="beef5678",
            head={"ref": "review/v0.2.0rc65+west-task-2e"},
        )
        with _patch_rest(payload=payload):
            r = resolve_merged_pr("https://github.com/o/r/pull/9")
        assert r.head_ref == "review/v0.2.0rc65+west-task-2e"

    def test_release_branch_name_resolves_back_to_the_frozen_artifact(self, tmp_path):
        """release → branch name → merge locates the same artifact file."""
        from imas_codex.standard_names.catalog_release import (
            _format_tag,
            _freeze_review_artifact,
        )
        from imas_codex.standard_names.promote import approval_tag_name
        from imas_codex.standard_names.sources_manifest import load_names_file

        tag = _format_tag(0, 2, 0, 65, build="west_task_2e")
        artifact = _freeze_review_artifact(
            tmp_path,
            rc_version=tag,
            names=["plasma_current"],
            minted_from="west_task_2e.yaml",
            unmatched=[],
        )
        branch = f"review/{tag}"
        rc = approval_tag_name(branch)
        # The branch name alone yields the full version, so the artifact is
        # located without recomputing the label from any manifest.
        assert rc == tag == "v0.2.0rc65+west-task-2e"
        located = tmp_path / f"{rc}.sn_names.yaml"
        assert located == artifact
        assert load_names_file(located) == ["plasma_current"]

        # rc_version carries the FULL version; batch_label is only the part
        # after the '+'. The two must not be conflated — merge resolves the
        # artifact from the former and reports batch identity from the latter.
        import yaml as _yaml

        doc = _yaml.safe_load(artifact.read_text(encoding="utf-8"))
        assert doc["rc_version"] == "v0.2.0rc65+west-task-2e"
        assert doc["batch_label"] == "west-task-2e"
        assert doc["rc_version"].split("+", 1)[1] == doc["batch_label"]


class TestFetchPrEvidenceOverRest:
    """Approval evidence is assembled from four REST reads, never from ``gh``.

    The CLI returned description, conversation and commits in one call; REST
    splits them across the pull, issue-comment, review and commit endpoints,
    so the assembler is what keeps the evidence record's shape stable for the
    summary synthesizer downstream.
    """

    def _transport(self, responses):
        def fake_call(method, path, *, payload=None, token=None):
            for fragment, status, body in responses:
                if fragment in path:
                    return status, body
            return 404, {"message": f"unrouted {path}"}

        return patch("imas_codex.graph.ghcr.github_api_call", fake_call)

    def test_assembles_body_conversation_and_commits(self):
        from imas_codex.standard_names.promote import fetch_pr_evidence

        responses = [
            ("/issues/7/comments", 200, [{"user": {"login": "rev"}, "body": "ok"}]),
            (
                "/pulls/7/reviews",
                200,
                [{"user": {"login": "rev"}, "body": "approve", "state": "APPROVED"}],
            ),
            (
                "/pulls/7/commits",
                200,
                [{"sha": "c1", "commit": {"message": "publish batch\n\nwhy"}}],
            ),
            ("/pulls/7", 200, {"body": "PR description"}),
        ]
        with self._transport(responses):
            evidence = fetch_pr_evidence("https://github.com/o/r/pull/7")

        assert evidence["body"] == "PR description"
        assert evidence["comments"][0]["author"]["login"] == "rev"
        assert evidence["reviews"][0]["state"] == "APPROVED"
        assert evidence["commits"][0]["oid"] == "c1"
        assert evidence["commits"][0]["messageHeadline"] == "publish batch"
        assert evidence["commits"][0]["messageBody"] == "why"

    def test_a_rejected_read_returns_empty_and_never_raises(self):
        from imas_codex.standard_names.promote import fetch_pr_evidence

        with self._transport([("/pulls/7", 401, {"message": "Bad credentials"})]):
            assert fetch_pr_evidence("https://github.com/o/r/pull/7") == {}

    def test_a_url_that_is_not_a_pull_request_returns_empty(self):
        from imas_codex.standard_names.promote import fetch_pr_evidence

        assert fetch_pr_evidence("not-a-url") == {}
