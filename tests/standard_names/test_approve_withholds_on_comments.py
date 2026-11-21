"""Approval must not treat unread review comments as assent."""

from __future__ import annotations

from unittest.mock import MagicMock, patch

from iter_codex.standard_names.promote import run_approval

_PR = {
    "catalog_pr_number": 12,
    "catalog_pr_url": "https://github.com/example/catalog/pull/12",
    "catalog_merge_commit_sha": "merge-sha",
}


def _approve(*, tmp_path, batch, review_evidence):
    graph = MagicMock()
    with (
        patch("iter_codex.standard_names.promote.read_pr_changes", return_value=[]),
        patch(
            "iter_codex.standard_names.promote._prepare_additive_catalog_delta",
            return_value=None,
        ),
        patch(
            "iter_codex.standard_names.promote.mark_catalog_name_approved",
            return_value=True,
        ) as mark,
    ):
        report = run_approval(
            isnc_dir=tmp_path,
            base_ref="base",
            batch=batch,
            review_evidence=review_evidence,
            gc=graph,
            **_PR,
        )
    return report, mark


def test_request_comment_withholds_every_untouched_identity(tmp_path):
    report, mark = _approve(
        tmp_path=tmp_path,
        batch=["pulse_duration", "plasma_current"],
        review_evidence={
            "comments": [{"body": "The physics definition needs clarification."}],
            "reviews": [],
            "review_comments": [],
        },
    )

    assert report.auto_approved == []
    assert [item["sn_id"] for item in report.blocked] == [
        "pulse_duration",
        "plasma_current",
    ]
    assert all(
        "unresolved reviewer comment" in item["reason"] for item in report.blocked
    )
    assert mark.call_count == 0


def test_line_anchored_comment_withholds_only_its_identity(tmp_path):
    catalog = tmp_path / "standard_names" / "equilibrium.yml"
    catalog.parent.mkdir()
    catalog.write_text(
        "- name: pulse_duration\n"
        "  kind: scalar\n"
        "  unit: s\n"
        "  description: duration\n\n"
        "- name: plasma_current\n"
        "  kind: scalar\n"
        "  unit: A\n"
        "  description: current\n",
        encoding="utf-8",
    )

    report, mark = _approve(
        tmp_path=tmp_path,
        batch=["pulse_duration", "plasma_current"],
        review_evidence={
            "comments": [],
            "reviews": [],
            "review_comments": [
                {
                    "body": "This definition is not dimensionally complete.",
                    "path": "standard_names/equilibrium.yml",
                    "line": 4,
                }
            ],
        },
    )

    assert report.auto_approved == ["plasma_current"]
    assert [item["sn_id"] for item in report.blocked] == ["pulse_duration"]
    assert mark.call_count == 1
    assert mark.call_args.args[0] == "plasma_current"


def test_silent_unchanged_batch_still_auto_approves(tmp_path):
    report, mark = _approve(
        tmp_path=tmp_path,
        batch=["pulse_duration", "plasma_current"],
        review_evidence={"comments": [], "reviews": [], "review_comments": []},
    )

    assert report.auto_approved == ["pulse_duration", "plasma_current"]
    assert report.blocked == []
    assert mark.call_count == 2
