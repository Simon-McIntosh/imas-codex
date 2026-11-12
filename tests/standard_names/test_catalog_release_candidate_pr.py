"""A whole-catalog candidate is delivered to the fork as a review request.

The candidate is the revision a reviewer reads before the catalog becomes
canonical, so it reaches the fork as a release branch plus one pull request
whose body carries markdown links rather than bare URLs. A candidate pointed
at the upstream catalog is refused before anything is pushed.
"""

from __future__ import annotations

import re
import subprocess
from pathlib import Path
from types import SimpleNamespace

import click
import pytest

from imas_codex.standard_names.catalog_release import run_release


def _git(*args: str, cwd: Path) -> subprocess.CompletedProcess[str]:
    return subprocess.run(
        ["git", *args], cwd=cwd, capture_output=True, text=True, check=True
    )


@pytest.fixture
def catalog_repo(tmp_path: Path) -> Path:
    repo = tmp_path / "isnc"
    repo.mkdir()
    _git("init", "-q", "-b", "main", cwd=repo)
    _git("config", "user.email", "t@t", cwd=repo)
    _git("config", "user.name", "t", cwd=repo)
    (repo / "README.md").write_text("x\n", encoding="utf-8")
    _git("add", "README.md", cwd=repo)
    _git("commit", "-qm", "init", cwd=repo)
    return repo


def _stand_in_remote(repo: Path, tmp_path: Path, *, name: str = "origin") -> Path:
    """A local bare repository standing in for the release remote."""
    remote = tmp_path / f"{name}.git"
    _git("init", "--bare", str(remote), cwd=tmp_path)
    _git("remote", "add", name, str(remote), cwd=repo)
    _git("push", "-q", name, "main", cwd=repo)
    return remote


def _staging(tmp_path: Path, *, export_report: bool = False) -> Path:
    staging = tmp_path / "staging"
    staging.mkdir()
    (staging / "catalog.yml").write_text("names: []\n", encoding="utf-8")
    if export_report:
        (staging / ".export_report.json").write_text("{}\n", encoding="utf-8")
    return staging


def _isolate_preflight(monkeypatch, tag: str = "v1.0.0rc1") -> None:
    monkeypatch.setattr(
        "imas_codex.standard_names.catalog_release._check_on_main", lambda _path: None
    )
    monkeypatch.setattr(
        "imas_codex.standard_names.catalog_release._check_clean_tree",
        lambda _path, **_kwargs: [],
    )
    monkeypatch.setattr(
        "imas_codex.standard_names.catalog_release._check_synced",
        lambda _path, _remote, **_kwargs: [],
    )
    monkeypatch.setattr(
        "imas_codex.standard_names.catalog_release.compute_next_version",
        lambda *_args, **_kwargs: (tag, tag.lstrip("v")),
    )


class _PublishingCatalog:
    """Copy the staging catalog into the checkout and commit it."""

    def __init__(self, *, export_report: bool = False) -> None:
        self.calls: list[dict] = []
        self.relpath = "RELEASED.md"
        self.export_report = export_report

    def __call__(self, *, isnc_path, **kwargs):
        self.calls.append(kwargs)
        path = Path(isnc_path)
        (path / self.relpath).write_text("catalog\n", encoding="utf-8")
        if self.export_report:
            (path / ".export_report.json").write_text("{}\n", encoding="utf-8")
        _git("add", self.relpath, cwd=path)
        if self.export_report:
            _git("add", ".export_report.json", cwd=path)
        _git("commit", "-qm", "release catalog", cwd=path)
        sha = _git("rev-parse", "HEAD", cwd=path).stdout.strip()
        return SimpleNamespace(errors=[], files_copied=1, commit_sha=sha)


def _bare_urls(body: str) -> list[str]:
    """URLs the body carries outside a markdown link target."""
    return re.findall(r"(?<!\]\()https?://[^\s)\]]+", body)


def test_a_whole_catalog_candidate_opens_one_review_pull_request(
    catalog_repo, tmp_path, monkeypatch
):
    remote = _stand_in_remote(catalog_repo, tmp_path)
    staging = _staging(tmp_path, export_report=True)
    publisher = _PublishingCatalog(export_report=True)
    _isolate_preflight(monkeypatch)
    requests: list[dict] = []

    def pr_creator(**kwargs):
        requests.append(kwargs)
        return 11, "https://github.com/reviewer/imas-standard-names/pull/11"

    report = run_release(
        catalog_repo,
        "catalog candidate",
        bump="minor",
        remote="origin",
        skip_export=True,
        staging_dir=staging,
        publisher=publisher,
        pr_creator=pr_creator,
        fork_owner="reviewer",
        upstream_repo="iterorganization/imas-standard-names",
    )

    assert report.errors == []
    assert len(requests) == 1, "a candidate opens one review request"
    request = requests[0]
    assert request["repo"] == "reviewer/imas-standard-names"
    assert request["head_owner"] == "reviewer"
    assert request["base"] == "main"
    assert request["branch"] == "release/v1.0.0rc1"
    assert request["title"] == "catalog candidate"

    assert "](https://" in request["body"], "the body's links are markdown links"
    assert _bare_urls(request["body"]) == [], "no bare URL survives in the body"
    assert (
        "https://github.com/reviewer/imas-standard-names/blob/release/v1.0.0rc1"
        "/.export_report.json"
    ) in request["body"], "the review body routes to the exclusion report"

    assert report.pr_number == 11
    assert report.pr_url.endswith("/pull/11")
    assert report.branch == "release/v1.0.0rc1"

    branch_head = _git(
        "rev-parse", "refs/heads/release/v1.0.0rc1", cwd=remote
    ).stdout.strip()
    assert branch_head == report.commit_sha, "the request names the released commit"
    tags = _git(
        "ls-remote", "--tags", "origin", "refs/tags/v1.0.0rc1", cwd=catalog_repo
    )
    assert tags.stdout.strip(), "the candidate still carries its tag"


def test_a_whole_catalog_candidate_refuses_an_upstream_remote(
    catalog_repo, tmp_path, monkeypatch
):
    upstream = _stand_in_remote(catalog_repo, tmp_path, name="upstream")
    before = _git("for-each-ref", "--format=%(refname)", cwd=upstream).stdout
    staging = _staging(tmp_path)
    publisher = _PublishingCatalog()
    _isolate_preflight(monkeypatch)

    def pr_creator(**_kwargs):
        raise AssertionError("a refused candidate must open no pull request")

    report = run_release(
        catalog_repo,
        "catalog candidate",
        bump="minor",
        remote="upstream",
        skip_export=True,
        staging_dir=staging,
        publisher=publisher,
        pr_creator=pr_creator,
    )

    assert len(report.errors) == 1, report.errors
    assert "must target the fork" in report.errors[0]
    assert "upstream" in report.errors[0]
    assert publisher.calls == [], "a refused candidate publishes nothing"
    assert _git("for-each-ref", "--format=%(refname)", cwd=upstream).stdout == before, (
        "nothing was pushed"
    )
    assert _git("tag", "-l", cwd=catalog_repo).stdout.strip() == "", "no tag was cut"
    assert report.pr_number is None


def test_the_cli_target_guard_refuses_upstream_for_a_candidate(
    catalog_repo, monkeypatch
):
    from imas_codex.cli.release import _check_release_target

    _git(
        "remote",
        "add",
        "origin",
        "https://github.com/reviewer/imas-standard-names.git",
        cwd=catalog_repo,
    )
    _git(
        "remote",
        "add",
        "upstream",
        "https://github.com/iterorganization/imas-standard-names.git",
        cwd=catalog_repo,
    )
    monkeypatch.chdir(catalog_repo)

    _check_release_target("origin", final=False)
    _check_release_target("upstream", final=True)

    with pytest.raises(click.ClickException, match="must target the fork"):
        _check_release_target("upstream", final=False)
    with pytest.raises(click.ClickException, match="must target upstream"):
        _check_release_target("origin", final=True)


def test_a_final_release_still_opens_its_pull_request_upstream(
    catalog_repo, tmp_path, monkeypatch
):
    """The final path keeps its own transport: upstream PR from a fork branch."""
    _stand_in_remote(catalog_repo, tmp_path)
    staging = _staging(tmp_path)
    _isolate_preflight(monkeypatch, tag="v1.0.0")
    requests: list[dict] = []

    def pr_creator(**kwargs):
        requests.append(kwargs)
        return 3, "https://github.com/iterorganization/imas-standard-names/pull/3"

    report = run_release(
        catalog_repo,
        "catalog release",
        final=True,
        remote="origin",
        skip_export=True,
        staging_dir=staging,
        publisher=_PublishingCatalog(),
        pr_creator=pr_creator,
        fork_owner="reviewer",
        upstream_repo="iterorganization/imas-standard-names",
    )

    assert report.errors == []
    assert len(requests) == 1
    assert requests[0]["repo"] == "iterorganization/imas-standard-names"
    assert requests[0]["head_owner"] == "reviewer"
    assert report.branch == "release/v1.0.0"
