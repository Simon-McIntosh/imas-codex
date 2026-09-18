"""A fold-back refusal has to reach the pull request the reviewer reads.

The reviewer's only surface is the request. When the fold-back refuses an
edit, the catalog has already merged and the reviewer has moved on, so a
refusal that stays in the run log is indistinguishable from an approval: the
request reads as accepted and the graph silently disagrees with the catalog.

These cases drive ``run_approval`` through one refused promotion and one that
succeeds, with the review scorer, the edit applier and the graph all stubbed,
and assert what reaches the request, through which transport. Neither case
needs a live graph or a network.
"""

from __future__ import annotations

from types import SimpleNamespace
from unittest.mock import patch

from imas_codex.standard_names import promote as approval_mod
from imas_codex.standard_names.promote import ApprovalChange, run_approval

PR_NUMBER = 7
PR_URL = "https://github.com/x/y/pull/7"
PR_SHA = "abc123"
REFUSAL_REASON = "catalog lifecycle promotion preconditions were not met"


class _FakeGraph:
    """Both read paths answer "present": one row, and the row counts as one."""

    def query(self, _cypher: str, **_params):
        return [{"n": 1}]


class _Transport:
    """Records every REST call and answers each one with a created comment."""

    def __init__(self, status: int = 201) -> None:
        self.calls: list[dict] = []
        self.status = status

    def __call__(self, method, path, *, payload=None, token=None):
        self.calls.append({"method": method, "path": path, "payload": payload})
        return self.status, {"id": 1}


def _drive(mark_approved: bool, transport: _Transport) -> object:
    """Run one approval of a single edited name, with the transport stubbed."""
    change = ApprovalChange(
        sn_id="__refusaltest__", axis="docs", new_value="new", old_value="old"
    )
    with (
        patch.object(approval_mod, "read_pr_changes", lambda *a, **k: [change]),
        patch.object(
            approval_mod,
            "apply_edit",
            lambda **k: SimpleNamespace(blocked=None, successor=None, run_id=None),
        ),
        patch.object(approval_mod, "_score_proposal", lambda *a, **k: 0.95),
        patch.object(approval_mod, "_apply_passing_review", lambda *a, **k: "accepted"),
        patch.object(approval_mod, "mark_catalog_name_approved", lambda *a, **k: mark_approved),
        patch("imas_codex.graph.ghcr.github_api_call", transport),
    ):
        return run_approval(
            isnc_dir="/unused",
            base_ref="main",
            gc=_FakeGraph(),
            catalog_pr_number=PR_NUMBER,
            catalog_pr_url=PR_URL,
            catalog_merge_commit_sha=PR_SHA,
            catalog_reviewer_actor="reviewer",
        )


def test_refused_promotion_posts_the_refusal_to_the_request() -> None:
    transport = _Transport()
    report = _drive(mark_approved=False, transport=transport)

    assert report.promotion_refused, "the guard refusal was not recorded"

    assert len(transport.calls) == 1, (
        "a refusal must post exactly one notice; got "
        f"{[c['path'] for c in transport.calls]}"
    )
    call = transport.calls[0]
    assert call["method"] == "POST"
    assert call["path"] == f"/repos/x/y/issues/{PR_NUMBER}/comments"

    body = call["payload"]["body"]
    assert REFUSAL_REASON in body, (
        "the reviewer-visible body must carry the refusal reason"
    )
    assert "__refusaltest__" in body, (
        "the reviewer must be told which identity was refused"
    )


def test_successful_promotion_posts_no_refusal() -> None:
    transport = _Transport()
    report = _drive(mark_approved=True, transport=transport)

    assert report.accepted, "the successful promotion was not recorded"
    assert not report.promotion_refused
    assert transport.calls == [], (
        "a promotion that succeeded must write nothing to the request; got "
        f"{[c['path'] for c in transport.calls]}"
    )