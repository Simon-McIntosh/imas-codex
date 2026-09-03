"""Catalog approval records the human identity from merged-PR evidence."""

from types import SimpleNamespace
from unittest.mock import patch

import pytest

from imas_codex.standard_names.promote import (
    ApprovalChange,
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


class RecordingGraph:
    def __init__(self) -> None:
        self.calls: list[tuple[str, dict]] = []

    def query(self, statement: str, **params):
        self.calls.append((statement, params))
        if "APPROVAL_MATCH_BY_ID" in statement:
            return [{"n": 1}]
        if "catalog_reviewer_actor = $reviewer_actor" in statement:
            return [{"id": params["name"]}]
        return []


def test_accepted_catalog_edit_persists_actor_from_pull_request_evidence():
    # REST reports a merged pull request as closed with a merge record beside
    # it, so the merged disposition is carried by ``merged``, not by ``state``.
    payload = {
        "number": 12,
        "html_url": "https://github.com/o/r/pull/12",
        "state": "closed",
        "merged": True,
        "merge_commit_sha": "abc123",
        "user": {"login": "physics-reviewer"},
        "head": {"ref": "review/catalog"},
        "base": {"ref": "main"},
    }
    with patch(
        "imas_codex.graph.ghcr.github_api_call",
        return_value=(200, payload),
    ):
        evidence = resolve_merged_pr("https://github.com/o/r/pull/12")

    change = ApprovalChange(
        sn_id="electron_temperature",
        axis="name",
        new_value="electron_thermal_energy",
    )
    graph = RecordingGraph()
    with (
        patch(
            "imas_codex.standard_names.promote.read_pr_changes",
            return_value=[change],
        ),
        patch(
            "imas_codex.standard_names.promote.apply_edit",
            return_value=SimpleNamespace(
                blocked=None,
                successor="electron_thermal_energy",
                run_id="catalog-edit",
            ),
        ),
        patch("imas_codex.standard_names.promote._score_proposal", return_value=0.95),
        patch(
            "imas_codex.standard_names.promote._apply_passing_review",
            return_value="accepted",
        ),
    ):
        report = run_approval(
            isnc_dir="/unused",
            base_ref="abc123^1",
            catalog_pr_number=evidence.number,
            catalog_pr_url=evidence.url,
            catalog_merge_commit_sha=evidence.merge_commit,
            gc=graph,
        )

    approval_writes = [
        params
        for statement, params in graph.calls
        if "catalog_reviewer_actor = $reviewer_actor" in statement
    ]
    assert report.accepted == ["electron_thermal_energy"]
    assert evidence.reviewer_actor == "physics-reviewer"
    assert approval_writes == [
        {
            "name": "electron_thermal_energy",
            "pr_number": 12,
            "pr_url": "https://github.com/o/r/pull/12",
            "merge_commit": "abc123",
            "reviewer_actor": "physics-reviewer",
            "editorial_outcome": "content_edit",
            "change_reason": "Catalog PR 12 recorded the content_edit editorial outcome.",
            "change_origin": "catalog_promotion",
        }
    ]
    assert "edit_origin" not in approval_writes[0]
