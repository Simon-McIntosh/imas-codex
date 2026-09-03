from __future__ import annotations

import subprocess
from pathlib import Path
from types import SimpleNamespace

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


def _focus_file_with_ledger(tmp_path: Path) -> Path:
    """A focus manifest with a committed exclusion ledger beside it.

    The ledger's presence is what triggers the exclusion-ledger paragraph on
    a synthesized body; committing it inside ``tmp_path`` gives it a
    resolvable blob address.
    """
    _git("init", "-b", "main", cwd=tmp_path)
    _git("config", "user.email", "test@example.com", cwd=tmp_path)
    _git("config", "user.name", "Test User", cwd=tmp_path)
    _git(
        "remote",
        "add",
        "origin",
        "https://github.com/west-owner/west-manifests.git",
        cwd=tmp_path,
    )
    focus = tmp_path / "batch.yaml"
    focus.write_text(
        "kind: sn_names\n"
        "schema_version: 1\n"
        "name: authored-body-check\n"
        "names:\n"
        "  - plasma_current\n",
        encoding="utf-8",
    )
    ledger = tmp_path / "batch.exclusions.json"
    ledger.write_text('{"excluded": []}\n', encoding="utf-8")
    _git("add", "batch.yaml", "batch.exclusions.json", cwd=tmp_path)
    _git("commit", "-m", "add batch manifest and exclusion ledger", cwd=tmp_path)
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
    def __init__(self) -> None:
        self.created_body: str | None = None
        self.updated_body: str | None = None

    def create_pull_request(self, **kwargs):
        self.created_body = kwargs["body"]
        return 17, f"https://github.com/{kwargs['repo']}/pull/17"

    def update_pull_request_body(self, **kwargs) -> None:
        assert self.created_body is not None
        self.updated_body = kwargs["body"]

    def read_pull_request_body(self, **_kwargs) -> str:
        assert self.updated_body is not None
        return self.updated_body


def _release(
    tmp_path: Path, client: MockGitHubClient, *, focus, pr_title=None, pr_body=None
):
    checkout = _catalog_checkout(tmp_path)
    return run_review_release(
        checkout,
        focus,
        "authored body review",
        staging_dir=tmp_path / "staging",
        bump="minor",
        reviews_dir=tmp_path / "reviews",
        exporter=_exporter,
        publisher=_publisher,
        github_client=client,
        upstream_repo="review-owner/example-catalog",
        fork_owner="review-owner",
        pr_target="fork",
        pr_title=pr_title,
        pr_body=pr_body,
    )


def test_authored_body_reaches_pr_unchanged_apart_from_preview_line(tmp_path):
    focus = _focus_file_with_ledger(tmp_path)
    client = MockGitHubClient()
    authored_body = "Hand-written review notes.\n\nNo machinery should touch this."

    report = _release(
        tmp_path,
        client,
        focus=focus,
        pr_title="authored review",
        pr_body=authored_body,
    )

    assert report.errors == []
    assert client.created_body == authored_body
    assert client.updated_body is not None
    assert client.updated_body.startswith(authored_body.rstrip())
    added = client.updated_body[len(authored_body.rstrip()) :]
    assert added.count("Preview:") == 1
    assert "Excluded source paths" not in client.updated_body


def test_synthesized_body_still_gets_exclusion_ledger_paragraph(tmp_path):
    focus = _focus_file_with_ledger(tmp_path)
    client = MockGitHubClient()

    report = _release(tmp_path, client, focus=focus)

    assert report.errors == []
    assert client.created_body is not None
    assert "Excluded source paths" in client.created_body
