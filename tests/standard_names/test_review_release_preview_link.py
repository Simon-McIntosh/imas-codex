from __future__ import annotations

import subprocess
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

from click.testing import CliRunner

from imas_codex.cli.sn import sn
from imas_codex.standard_names.catalog_release import run_review_release


def _git(*args: str, cwd: Path) -> subprocess.CompletedProcess[str]:
    return subprocess.run(
        ["git", *args],
        cwd=cwd,
        capture_output=True,
        text=True,
        check=True,
    )


def _catalog_checkout(tmp_path: Path) -> Path:
    bare = tmp_path / "origin.git"
    _git("init", "--bare", "-b", "main", str(bare), cwd=tmp_path)
    checkout = tmp_path / "catalog"
    checkout.mkdir()
    _git("init", "-b", "main", cwd=checkout)
    _git("config", "user.email", "test@example.com", cwd=checkout)
    _git("config", "user.name", "Test User", cwd=checkout)
    _git("remote", "add", "origin", str(bare), cwd=checkout)
    (checkout / "README.md").write_text("catalog\n", encoding="utf-8")
    _git("add", "README.md", cwd=checkout)
    _git("commit", "-m", "initial catalog", cwd=checkout)
    _git("push", "origin", "main", cwd=checkout)
    return checkout


def _focus_file(tmp_path: Path) -> Path:
    focus = tmp_path / "batch.yaml"
    focus.write_text(
        "kind: sn_names\n"
        "schema_version: 1\n"
        "name: preview-check\n"
        "names:\n"
        "  - plasma_current\n",
        encoding="utf-8",
    )
    return focus


def _exporter(**kwargs):
    staging = Path(kwargs["staging_dir"])
    (staging / "standard_names").mkdir(parents=True, exist_ok=True)
    (staging / "catalog.yml").write_text("catalog_name: test\n", encoding="utf-8")
    (staging / "standard_names" / "equilibrium.yml").write_text(
        "- name: plasma_current\n  unit: A\n",
        encoding="utf-8",
    )
    return SimpleNamespace(exported_count=1)


def _publisher(**kwargs):
    checkout = Path(kwargs["isnc_path"])
    (checkout / "catalog.yml").write_text("catalog_name: test\n", encoding="utf-8")
    _git("add", "catalog.yml", cwd=checkout)
    _git("commit", "-m", "publish review catalog", cwd=checkout)
    return SimpleNamespace(errors=[], commit_sha="created", files_copied=1)


class MockGitHubClient:
    def __init__(self, *, return_written_body: bool = True) -> None:
        self.return_written_body = return_written_body
        self.created_body: str | None = None
        self.updated_body: str | None = None

    def create_pull_request(self, **kwargs):
        self.created_body = kwargs["body"]
        return 17, f"https://github.com/{kwargs['repo']}/pull/17"

    def update_pull_request_body(self, **kwargs) -> None:
        assert self.created_body is not None
        self.updated_body = kwargs["body"]

    def read_pull_request_body(self, **_kwargs) -> str:
        if self.return_written_body:
            assert self.updated_body is not None
            return self.updated_body
        return self.created_body or ""


def _release(tmp_path: Path, client: MockGitHubClient):
    checkout = _catalog_checkout(tmp_path)
    return run_review_release(
        checkout,
        _focus_file(tmp_path),
        "preview review",
        staging_dir=tmp_path / "staging",
        bump="minor",
        reviews_dir=tmp_path / "reviews",
        exporter=_exporter,
        publisher=_publisher,
        github_client=client,
        upstream_repo="review-owner/example-catalog",
        fork_owner="review-owner",
        pr_target="fork",
    )


def test_release_writes_and_reads_back_exact_pr_preview_address(tmp_path):
    client = MockGitHubClient()

    report = _release(tmp_path, client)

    expected = "https://review-owner.github.io/example-catalog/pr-17/"
    assert report.errors == []
    assert client.created_body is not None
    assert expected not in client.created_body
    assert client.updated_body is not None
    assert expected in client.updated_body


def test_release_names_missing_preview_link_invariant(tmp_path):
    report = _release(tmp_path, MockGitHubClient(return_written_body=False))

    assert report.errors == [
        "ReviewPreviewLinkInvariantError: read-back body for "
        "review-owner/example-catalog#17 lacks exact preview address "
        "https://review-owner.github.io/example-catalog/pr-17/"
    ]


def test_release_command_exits_nonzero_for_missing_preview_link(tmp_path):
    focus = _focus_file(tmp_path)
    failure = SimpleNamespace(
        errors=["ReviewPreviewLinkInvariantError: missing exact preview address"]
    )

    with patch(
        "imas_codex.standard_names.catalog_release.run_review_release",
        return_value=failure,
    ):
        result = CliRunner().invoke(
            sn,
            [
                "release",
                "--batch",
                str(focus),
                "--isnc",
                str(tmp_path),
                "--dry-run",
                "-m",
                "preview review",
            ],
        )

    assert result.exit_code == 1
    assert "ReviewPreviewLinkInvariantError" in result.output
