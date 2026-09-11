"""The per-reason exclusion breakdown is reachable from a published cut.

The exporter writes the complete per-reason accounting into
``.export_report.json`` on every export path, and the publish path now
copies that file beside the published ``catalog.yml`` so it rides the
published commit. This module pins the other half of the guarantee: the
composed review body carries a descriptive Markdown link to the report's
derived address, and a cut whose report never reached the committed
checkout refuses the request rather than publishing a link to nothing.
"""

from __future__ import annotations

import json
import re
import subprocess
from pathlib import Path
from types import SimpleNamespace

import pytest

from imas_codex.standard_names.catalog_release import (
    ExportReportLinkError,
    body_with_export_report_link,
    export_report_blob_url,
    run_review_release,
)

_PR_TARGET = {
    "upstream_repo": "acme-owner/catalog",
    "fork_owner": "acme-fork",
}


def _git(*args, cwd):
    return subprocess.run(
        ["git", *args], cwd=cwd, capture_output=True, text=True, check=True
    )


@pytest.fixture
def isnc_repo(tmp_path):
    """A local catalog checkout on 'main' with a bare 'origin' remote."""
    bare = tmp_path / "origin.git"
    _git("init", "--bare", "-b", "main", str(bare), cwd=tmp_path)
    work = tmp_path / "isnc"
    work.mkdir()
    _git("init", "-b", "main", cwd=work)
    _git("config", "user.email", "t@t", cwd=work)
    _git("config", "user.name", "t", cwd=work)
    _git("remote", "add", "origin", str(bare), cwd=work)
    (work / "README.md").write_text("isnc\n")
    _git("add", "README.md", cwd=work)
    _git("commit", "-m", "init", cwd=work)
    _git("push", "origin", "main", cwd=work)
    return work


def _write_names_focus(tmp_path, *, name="west-task-2e", filename="batch.yaml"):
    path = tmp_path / filename
    path.write_text(
        "kind: sn_names\n"
        "schema_version: 1\n"
        f"name: {name}\n"
        "names:\n"
        "  - poloidal_flux\n"
        "  - plasma_current\n",
        encoding="utf-8",
    )
    return path


def _write_export_report(staging: Path) -> None:
    """Write a report with a distinctive per-reason marker into staging."""
    (staging / ".export_report.json").write_text(
        json.dumps(
            {
                "emitted_identities": ["plasma_current"],
                "exclusion_ledger": [
                    {
                        "reason": "name_not_accepted",
                        "count": 1,
                        "identities": ["poloidal_flux"],
                    }
                ],
            }
        ),
        encoding="utf-8",
    )


def _stub_exporter(record):
    def exporter(*, staging_dir, force, review_batch, **kw):
        record["review_batch"] = review_batch
        sd = Path(staging_dir)
        (sd / "standard_names").mkdir(parents=True, exist_ok=True)
        (sd / "standard_names" / "equilibrium.yml").write_text(
            "- name: plasma_current\n  unit: A\n", encoding="utf-8"
        )
        (sd / "catalog.yml").write_text(
            "catalog_name: t\ncandidate_count: 2\npublished_count: 1\n",
            encoding="utf-8",
        )
        # The export report must always be written; the publisher decides
        # whether the cut actually carries it.
        _write_export_report(sd)
        return SimpleNamespace(exported_count=len(review_batch))

    return exporter


def _stub_publisher(copy_report: bool):
    def publisher(*, staging_dir, isnc_path, push, allow_dirty):
        staging = Path(staging_dir)
        checkout = Path(isnc_path)
        (checkout / "catalog.yml").write_text(
            (staging / "catalog.yml").read_text(encoding="utf-8"),
            encoding="utf-8",
        )
        paths = ["catalog.yml"]
        if copy_report:
            (checkout / ".export_report.json").write_text(
                (staging / ".export_report.json").read_text(encoding="utf-8"),
                encoding="utf-8",
            )
            paths.append(".export_report.json")
        _git("add", *paths, cwd=checkout)
        _git("commit", "-m", "publish", cwd=checkout)
        return SimpleNamespace(
            errors=[], commit_sha="deadbeef", files_copied=len(paths)
        )

    return publisher


def _capture_pr(bodies, calls):
    def pr_creator(*, branch, base, title, body, repo, head_owner):
        calls["called"] = True
        bodies.append(body)
        return 42, f"https://github.com/{repo}/pull/42"

    return pr_creator


def _run_cut(isnc_repo, tmp_path, *, copy_report: bool, record: dict) -> tuple:
    focus = _write_names_focus(tmp_path)
    reviews = tmp_path / "reviews"
    pr_bodies: list[str] = []
    pr_calls = {"called": False}
    report = run_review_release(
        isnc_repo,
        focus,
        "Review batch demo",
        staging_dir=tmp_path / "staging",
        bump="minor",
        reviews_dir=reviews,
        exporter=_stub_exporter(record),
        publisher=_stub_publisher(copy_report=copy_report),
        pr_creator=_capture_pr(pr_bodies, pr_calls),
        dd_gap_reader=lambda **_kwargs: [],
        **_PR_TARGET,
    )
    return report, pr_bodies, pr_calls


def _bare_urls(text: str) -> list[str]:
    """URLs outside Markdown link brackets — the bare form a body must not use."""
    stripped = re.sub(r"\[[^\]]*\]\(([^)]*)\)", "", text)
    return re.findall(r"https?://\S+", stripped)


def test_publish_commits_report_and_body_links_its_derived_address(isnc_repo, tmp_path):
    """A cut that delivers the report links it from the composed body."""
    exporter_record: dict = {}
    report, pr_bodies, pr_calls = _run_cut(
        isnc_repo, tmp_path, copy_report=True, record=exporter_record
    )

    assert report.errors == []
    assert pr_calls["called"] is True
    # The report rode the published commit on the review branch — the exact
    # tree the derived blob address dereferences to — carrying its marker.
    # The shared checkout returns to main after the cut, so the reachable
    # copy is on the branch, not in the post-run working tree.
    committed_report = _git(
        "show", f"{report.branch}:.export_report.json", cwd=isnc_repo
    ).stdout
    assert '"name_not_accepted"' in committed_report
    assert json.loads(committed_report)["exclusion_ledger"][0]["reason"] == (
        "name_not_accepted"
    )
    # The composed body links that same report at its derived address, as a
    # descriptive Markdown link naming its destination.
    body = pr_bodies[0]
    derived = export_report_blob_url(
        fork_owner="acme-fork",
        upstream_repo="acme-owner/catalog",
        branch=report.branch,
    )
    assert derived == (
        f"https://github.com/acme-fork/catalog/blob/{report.branch}/.export_report.json"
    )
    assert f"[export exclusion report (.export_report.json)]({derived})" in body, (
        "the body must link the report text with a descriptive Markdown link"
    )
    assert _bare_urls(body) == []


def test_body_with_export_report_link_requires_the_published_report(tmp_path):
    """Composing a body that links an unpublished report refuses instead."""
    checkout = tmp_path / "catalog"
    checkout.mkdir()

    with pytest.raises(ExportReportLinkError, match=r"\.export_report\.json"):
        body_with_export_report_link(
            "Review content.",
            checkout=checkout,
            fork_owner="acme-fork",
            upstream_repo="acme-owner/catalog",
            branch="review/v0.1.0rc1+west-task-2e",
        )

    # Present: the body gains a descriptive Markdown link, no bare URL.
    (checkout / ".export_report.json").write_text("{}", encoding="utf-8")
    composed = body_with_export_report_link(
        "Review content.",
        checkout=checkout,
        fork_owner="acme-fork",
        upstream_repo="acme-owner/catalog",
        branch="review/v0.1.0rc1+west-task-2e",
    )
    assert (
        "[export exclusion report (.export_report.json)]"
        "(https://github.com/acme-fork/catalog/"
        "blob/review/v0.1.0rc1+west-task-2e/.export_report.json)" in composed
    )
    assert _bare_urls(composed) == []


def test_report_missing_from_checkout_refuses_the_request(isnc_repo, tmp_path):
    """A cut whose report never reached the checkout cannot open a request."""
    exporter_record: dict = {}
    report, pr_bodies, pr_calls = _run_cut(
        isnc_repo, tmp_path, copy_report=False, record=exporter_record
    )

    assert not (isnc_repo / ".export_report.json").exists()
    assert pr_calls["called"] is False
    assert pr_bodies == []
    assert report.pr_number is None
    assert any("ExportReportLinkError" in error for error in report.errors), (
        report.errors
    )
