"""Tests for ``sn merge`` — fold a reviewed catalog PR back into the ledger.

All graph interaction is MOCKED — these tests never touch Neo4j. They pin
the orchestration contract of :func:`run_merge`:

* a human catalog edit is re-attached like ``sn edit`` (candidate + reason,
  ``origin='human'``, ``refine=False``) via :func:`apply_edit`;
* the attached proposal is scored by the FULL review pipeline, but the
  refine pools are NEVER invoked (a human-reviewed wording must not be
  silently rewritten);
* a review score at or above threshold ACCEPTS (the name reaches the
  accepted state via ``persist_reviewed_*``);
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
from imas_codex.standard_names.merge import (
    MergeChange,
    MergeReport,
    read_pr_changes,
    run_merge,
)

MERGE = "imas_codex.standard_names.merge"


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


class TestRunMergeAcceptPath:
    def test_docs_edit_attached_like_sn_edit_with_reason(self):
        change = MergeChange(
            sn_id="electron_temperature",
            axis="docs",
            new_value="Electron temperature. Revised by reviewer.",
            old_value="Electron temperature.",
        )
        gc = _gc_exists(True)
        with (
            patch(f"{MERGE}.read_pr_changes", return_value=[change]),
            patch(f"{MERGE}.apply_edit") as m_apply,
            patch(f"{MERGE}._score_proposal", return_value=0.93) as m_score,
            patch(f"{MERGE}.persist_reviewed_docs", return_value="accepted") as m_docs,
            patch(f"{MERGE}.persist_reviewed_name") as m_name,
        ):
            m_apply.return_value = _edit_plan(target="electron_temperature")
            report = run_merge(
                isnc_dir="/tmp/isnc", base_ref="origin/main", threshold=0.85, gc=gc
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

        # Review ran, then accept via the docs persist path.
        m_score.assert_called_once()
        m_docs.assert_called_once()
        m_name.assert_not_called()
        assert "electron_temperature" in report.accepted
        assert report.threshold == 0.85

    def test_score_at_threshold_accepts(self):
        change = MergeChange(sn_id="ion_density", axis="docs", new_value="Ion density.")
        gc = _gc_exists(True)
        with (
            patch(f"{MERGE}.read_pr_changes", return_value=[change]),
            patch(f"{MERGE}.apply_edit", return_value=_edit_plan(target="ion_density")),
            patch(f"{MERGE}._score_proposal", return_value=0.85),
            patch(f"{MERGE}.persist_reviewed_docs", return_value="accepted") as m_docs,
        ):
            report = run_merge(isnc_dir="/x", base_ref="b", threshold=0.85, gc=gc)
        m_docs.assert_called_once()
        assert "ion_density" in report.accepted
        assert not report.quarantined


# ---------------------------------------------------------------------------
# Quarantine path (score below threshold)
# ---------------------------------------------------------------------------


class TestRunMergeQuarantinePath:
    def test_low_score_quarantines_not_accepted_not_refined(self):
        change = MergeChange(sn_id="ion_density", axis="docs", new_value="Bad docs.")
        gc = _gc_exists(True)
        with (
            patch(f"{MERGE}.read_pr_changes", return_value=[change]),
            patch(f"{MERGE}.apply_edit", return_value=_edit_plan(target="ion_density")),
            patch(f"{MERGE}._score_proposal", return_value=0.50),
            patch(f"{MERGE}.persist_reviewed_docs") as m_docs,
            patch(f"{MERGE}.persist_reviewed_name") as m_name,
        ):
            report = run_merge(isnc_dir="/x", base_ref="b", threshold=0.85, gc=gc)

        # NOT accepted — no promote path taken.
        m_docs.assert_not_called()
        m_name.assert_not_called()
        assert "ion_density" not in report.accepted

        # Quarantined + flagged.
        assert any(q["sn_id"] == "ion_density" for q in report.quarantined)
        # The existing quarantine signal is written to the graph.
        quarantine_queries = [
            c.args[0]
            for c in gc.query.call_args_list
            if c.args and "quarantined" in c.args[0]
        ]
        assert quarantine_queries, (
            "expected a gc.query setting validation_status='quarantined'"
        )
        assert "validation_status" in quarantine_queries[0]

    def test_full_review_runs_but_refine_never_invoked(self):
        """FULL review is exercised; NO refine pool is ever entered."""
        changes = [
            MergeChange(sn_id="a_name", axis="docs", new_value="Doc A."),
            MergeChange(sn_id="b_name", axis="docs", new_value="Doc B."),
        ]
        gc = _gc_exists(True)
        with (
            patch(f"{MERGE}.read_pr_changes", return_value=changes),
            patch(
                f"{MERGE}.apply_edit",
                side_effect=lambda **k: _edit_plan(target=k["target"]),
            ),
            patch(f"{MERGE}._score_proposal", side_effect=[0.9, 0.4]) as m_score,
            patch(f"{MERGE}.persist_reviewed_docs", return_value="accepted"),
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
            report = run_merge(isnc_dir="/x", base_ref="b", threshold=0.85, gc=gc)

        # Review scored every matched proposal…
        assert m_score.call_count == 2
        # …but NOT ONE refine claim/process was invoked.
        m_rn.assert_not_called()
        m_rd.assert_not_called()
        m_prn.assert_not_called()
        m_prd.assert_not_called()
        assert "a_name" in report.accepted
        assert any(q["sn_id"] == "b_name" for q in report.quarantined)


# ---------------------------------------------------------------------------
# Name edit → rename cascade
# ---------------------------------------------------------------------------


class TestRunMergeNameEdit:
    def test_name_edit_routes_through_rename_mode(self):
        change = MergeChange(
            sn_id="elongation",
            axis="name",
            new_value="elongation_of_closed_flux_surface",
        )
        gc = _gc_exists(True)
        with (
            patch(f"{MERGE}.read_pr_changes", return_value=[change]),
            patch(f"{MERGE}.apply_edit") as m_apply,
            patch(f"{MERGE}._score_proposal", return_value=0.95) as m_score,
            patch(f"{MERGE}.persist_reviewed_name", return_value="accepted") as m_name,
            patch(f"{MERGE}.persist_reviewed_docs") as m_docs,
        ):
            m_apply.return_value = _edit_plan(
                target="elongation",
                mode="rename",
                successor="elongation_of_closed_flux_surface",
            )
            report = run_merge(isnc_dir="/x", base_ref="b", threshold=0.85, gc=gc)

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


class TestRunMergeEdgeCases:
    def test_unmatched_id_is_reported_without_attaching(self):
        change = MergeChange(sn_id="ghost_name", axis="docs", new_value="x")
        gc = _gc_exists(False)
        with (
            patch(f"{MERGE}.read_pr_changes", return_value=[change]),
            patch(f"{MERGE}.apply_edit") as m_apply,
            patch(f"{MERGE}._score_proposal") as m_score,
        ):
            report = run_merge(isnc_dir="/x", base_ref="b", gc=gc)
        m_apply.assert_not_called()
        m_score.assert_not_called()
        assert "ghost_name" in report.unmatched

    def test_blocked_edit_is_recorded_and_not_scored(self):
        change = MergeChange(sn_id="frozen_name", axis="docs", new_value="x")
        gc = _gc_exists(True)
        with (
            patch(f"{MERGE}.read_pr_changes", return_value=[change]),
            patch(
                f"{MERGE}.apply_edit",
                return_value=_edit_plan(
                    target="frozen_name", blocked="target ineligible stage"
                ),
            ),
            patch(f"{MERGE}._score_proposal") as m_score,
            patch(f"{MERGE}.persist_reviewed_docs") as m_docs,
        ):
            report = run_merge(isnc_dir="/x", base_ref="b", gc=gc)
        m_score.assert_not_called()
        m_docs.assert_not_called()
        assert any(b["sn_id"] == "frozen_name" for b in report.blocked)

    def test_dry_run_attaches_nothing(self):
        change = MergeChange(sn_id="ion_density", axis="docs", new_value="x")
        gc = _gc_exists(True)
        with (
            patch(f"{MERGE}.read_pr_changes", return_value=[change]),
            patch(f"{MERGE}.apply_edit") as m_apply,
            patch(f"{MERGE}._score_proposal") as m_score,
            patch(f"{MERGE}.persist_reviewed_docs") as m_docs,
        ):
            report = run_merge(isnc_dir="/x", base_ref="b", dry_run=True, gc=gc)
        m_apply.assert_not_called()
        m_score.assert_not_called()
        m_docs.assert_not_called()
        assert report.dry_run is True
        assert isinstance(report, MergeReport)


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
