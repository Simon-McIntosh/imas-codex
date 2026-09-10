"""The release rehearsal reports what it does, and the body says what it links.

Verdicts at the time this module ships: four defects were recorded on the
review-batch release path, and at HEAD all four are closed or verified.

1. A dry run used to freeze a review roster, write into staging, and advance
   the release-candidate counter, because the freeze ran before the dry-run
   branch was consulted. The inert-rehearsal guard now returns before any
   staging, tag, or freeze work, and ``compute_next_version`` is pure, so a
   dry run reports the candidate it would take and writes nothing. This
   module pins that guarantee: no staging directory, no roster, no branch,
   and an unmoved candidate.
2. The rename cascade planner used to print descendant renames it never
   performed. The cascade now reports descendants as deferred (awaiting the
   root's acceptance) and leaves them untouched; that behaviour and its test
   live in the module that owns it and are not duplicated here.
3. The composed release body used to carry a bare preview URL and a
   hardcoded reviewing-guide address. The preview and exclusion-ledger links
   in the release path are now descriptive Markdown links whose addresses are
   derived from the checkout's own remotes. This module drives the body
   composition and asserts no bare URL survives and no address is spelled as
   a literal.
4. The catalog build on the review request had never been re-read after its
   body was fixed. The order-owning read is: query the owning repository
   listing with an explicit ``--repo``, never infer from a missing message.
   That read is a live operation recorded with this node's evidence, and the
   last review request's Validate Catalog and Catalog Site workflows are the
   receipt it produces.
"""

from __future__ import annotations

import re
import subprocess
from pathlib import Path
from types import SimpleNamespace

import pytest

from imas_codex.standard_names.catalog_release import (
    _review_preview_url,
    _write_and_verify_review_preview_link,
    body_with_exclusion_ledger_link,
    compute_next_version,
    run_review_release,
)

_PR_TARGET = {
    "upstream_repo": "example-org/example-catalog",
    "fork_owner": "example-fork",
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


def _stub_exporter(record):
    def exporter(*, staging_dir, force, review_batch, **kw):
        record["review_batch"] = review_batch
        sd = Path(staging_dir)
        (sd / "standard_names").mkdir(parents=True, exist_ok=True)
        (sd / "catalog.yml").write_text("catalog_name: t\n")
        return SimpleNamespace(exported_count=len(review_batch))

    return exporter


def _stub_publisher(isnc):
    def publisher(*, staging_dir, isnc_path, push, allow_dirty):
        (Path(isnc_path) / "catalog.yml").write_text("catalog_name: t\n")
        _git("add", "catalog.yml", cwd=isnc_path)
        _git("commit", "-m", "publish", cwd=isnc_path)
        return SimpleNamespace(errors=[], commit_sha="deadbeef", files_copied=1)

    return publisher


def _stub_pr():
    def pr_creator(*, branch, base, title, body, repo, head_owner):
        return 42, f"https://github.com/{repo}/pull/42"

    return pr_creator


# ── Defect 1: a dry run writes nothing and moves no candidate ─────────────


def test_rehearsal_writes_no_staging_no_roster_and_moves_no_candidate(
    isnc_repo, tmp_path
):
    """A rehearsal reports the candidate it would take and writes nothing."""
    focus = _write_names_focus(tmp_path)
    reviews = tmp_path / "reviews"
    reviews.mkdir()
    label = "west-task-2e"
    before_listing = sorted(p.name for p in reviews.iterdir())
    staging = tmp_path / "staging"
    before_candidate = compute_next_version(
        isnc_repo, "minor", final=False, build=label
    )
    exporter_record: dict = {}

    report = run_review_release(
        isnc_repo,
        focus,
        "Review batch demo",
        staging_dir=staging,
        bump="minor",
        dry_run=True,
        reviews_dir=reviews,
        exporter=_stub_exporter(exporter_record),
        publisher=_stub_publisher(isnc_repo),
        pr_creator=_stub_pr(),
        **_PR_TARGET,
    )

    assert report.errors == []
    # The rehearsal writes nothing that names a cut: no staging directory is
    # created, no roster is frozen, and no review branch appears.
    assert exporter_record == {}, "the export leg must not run on a dry run"
    assert not staging.exists()
    assert sorted(p.name for p in reviews.iterdir()) == before_listing
    assert "review/" not in _git("branch", cwd=isnc_repo).stdout
    # The RC counter does not move for a release that never happens.
    after_candidate = compute_next_version(isnc_repo, "minor", final=False, build=label)
    assert after_candidate == before_candidate
    # It still reports the candidate it would have taken.
    assert report.rc_version == before_candidate[0]
    assert report.pushed is False
    assert report.pr_number is None


# ── Defect 3: every release-body link is a descriptive, derived Markdown ──


class _RecordingGitHub:
    """Fake github client that returns exactly the body it was given."""

    def __init__(self):
        self.bodies: dict[tuple[str, int], str] = {}

    def update_pull_request_body(self, *, repo: str, number: int, body: str) -> None:
        self.bodies[(repo, number)] = body

    def read_pull_request_body(self, *, repo: str, number: int) -> str:
        return self.bodies.get((repo, number), "")

    def last_body(self) -> str:
        return next(reversed(self.bodies.values())) if self.bodies else ""


def _bare_urls(text: str) -> list[str]:
    """URLs outside Markdown link brackets — the bare form a body must not use."""
    stripped = re.sub(r"\[[^\]]*\]\(([^)]*)\)", "", text)
    return re.findall(r"https?://\S+", stripped)


def test_preview_link_is_descriptive_markdown_with_a_derived_address():
    client = _RecordingGitHub()
    repo = "acme-owner/catalog"
    pr_number = 42

    _write_and_verify_review_preview_link(
        client, repo=repo, pr_number=pr_number, body="Review content."
    )

    body = client.last_body()
    derived = _review_preview_url(repo, pr_number)
    assert derived == "https://acme-owner.github.io/catalog/pr-42/"
    assert f"Preview: [rendered catalog preview]({derived})" in body
    # The address is derived from the repo it targets, never spelled inline,
    # and no bare URL survives outside the Markdown brackets.
    assert _bare_urls(body) == []
    body_without_preview = body.split("Preview:", 1)[0]
    assert "https://" not in body_without_preview


def test_preview_read_back_mismatch_refuses():
    """The preview is verified by reading the body back, not by the absence
    of an error from the write."""

    class _DroppingGitHub:
        def update_pull_request_body(self, *, repo, number, body):
            pass  # never persisted

        def read_pull_request_body(self, *, repo, number):
            return "unchanged body without the preview"

    from imas_codex.standard_names.catalog_release import (
        ReviewPreviewLinkInvariantError,
    )

    with pytest.raises(ReviewPreviewLinkInvariantError, match="lacks exact"):
        _write_and_verify_review_preview_link(
            _DroppingGitHub(),
            repo="acme-owner/catalog",
            pr_number=7,
            body="Review content.",
        )


def test_exclusion_ledger_link_is_markdown_with_a_derived_address(tmp_path):
    """The ledger address follows the checkout's own remote, not a literal."""
    ledger_repo = tmp_path / "manifest-repo"
    ledger_repo.mkdir()
    _git("init", "-b", "main", cwd=ledger_repo)
    _git("config", "user.email", "t@t", cwd=ledger_repo)
    _git("config", "user.name", "t", cwd=ledger_repo)
    _git(
        "remote",
        "add",
        "origin",
        "git@github.com:acme-owner/catalog.git",
        cwd=ledger_repo,
    )
    ledger = ledger_repo / "west.exclusions.json"
    ledger.write_text('{"excluded": []}\n', encoding="utf-8")
    _git("add", "west.exclusions.json", cwd=ledger_repo)
    _git("commit", "-m", "ledger", cwd=ledger_repo)
    focus = ledger_repo / "west.yaml"
    focus.write_text("kind: sn_names\nname: demo\nnames: []\n", encoding="utf-8")
    before = "Review the batch and its exclusions."

    composed = body_with_exclusion_ledger_link(str(before), focus)

    # The ledger address is derived (owner/repo from the origin remote), so
    # the link target is the repo's own blob URL for the committed revision.
    links = re.findall(r"\[[^\]]*\]\(([^)]*)\)", composed)
    assert len(links) == 1
    assert links[0].startswith("https://github.com/acme-owner/catalog/blob/")
    assert "ledger of excluded source paths and their withholding data " in composed
    assert _bare_urls(composed) == []


def test_ledger_link_is_appended_once_not_duplicated(tmp_path):
    ledger_repo = tmp_path / "once"
    ledger_repo.mkdir()
    _git("init", "-b", "main", cwd=ledger_repo)
    _git("config", "user.email", "t@t", cwd=ledger_repo)
    _git("config", "user.name", "t", cwd=ledger_repo)
    _git(
        "remote",
        "add",
        "origin",
        "git@github.com:acme-owner/catalog.git",
        cwd=ledger_repo,
    )
    ledger = ledger_repo / "west.exclusions.json"
    ledger.write_text('{"excluded": []}\n', encoding="utf-8")
    _git("add", "west.exclusions.json", cwd=ledger_repo)
    _git("commit", "-m", "ledger", cwd=ledger_repo)
    focus = ledger_repo / "west.yaml"

    once = body_with_exclusion_ledger_link("body", focus)
    twice = body_with_exclusion_ledger_link(once, focus)

    assert twice == once, "a body that already names the ledger gains no second link"
