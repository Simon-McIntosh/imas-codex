"""Checkout-state guarantees for catalog review releases."""

from __future__ import annotations

import subprocess
from pathlib import Path
from types import SimpleNamespace

import pytest

from imas_codex.standard_names.catalog_release import run_review_release


def _git(*args: str, cwd: Path) -> subprocess.CompletedProcess[str]:
    return subprocess.run(
        ["git", *args], cwd=cwd, capture_output=True, text=True, check=True
    )


@pytest.fixture
def catalog_checkout(tmp_path: Path) -> Path:
    remote = tmp_path / "origin.git"
    _git("init", "--bare", "-b", "main", str(remote), cwd=tmp_path)
    checkout = tmp_path / "catalog"
    checkout.mkdir()
    _git("init", "-b", "main", cwd=checkout)
    _git("config", "user.email", "test@example.com", cwd=checkout)
    _git("config", "user.name", "Test User", cwd=checkout)
    _git("remote", "add", "origin", str(remote), cwd=checkout)
    (checkout / "README.md").write_text("catalog\n", encoding="utf-8")
    _git("add", "README.md", cwd=checkout)
    _git("commit", "-m", "Initial catalog", cwd=checkout)
    _git("push", "origin", "main", cwd=checkout)
    return checkout


def _focus_file(tmp_path: Path) -> Path:
    focus = tmp_path / "batch.yaml"
    focus.write_text(
        "kind: sn_names\n"
        "schema_version: 1\n"
        "name: checkout-test\n"
        "names:\n"
        "  - plasma_current\n",
        encoding="utf-8",
    )
    return focus


def _exporter(**kwargs: object) -> SimpleNamespace:
    staging = Path(str(kwargs["staging_dir"]))
    (staging / "standard_names").mkdir(parents=True, exist_ok=True)
    (staging / "catalog.yml").write_text("catalog_name: test\n", encoding="utf-8")
    return SimpleNamespace(exported_count=1)


def _publisher(**kwargs: object) -> SimpleNamespace:
    checkout = Path(str(kwargs["isnc_path"]))
    (checkout / "catalog.yml").write_text("catalog_name: test\n", encoding="utf-8")
    _git("add", "catalog.yml", cwd=checkout)
    _git("commit", "-m", "Publish catalog", cwd=checkout)
    return SimpleNamespace(errors=[], commit_sha="created", files_copied=1)


def test_review_release_restores_clean_main_with_remote_refs(
    catalog_checkout: Path, tmp_path: Path
) -> None:
    main_head = _git("rev-parse", "main", cwd=catalog_checkout).stdout.strip()

    report = run_review_release(
        catalog_checkout,
        _focus_file(tmp_path),
        "Review catalog batch",
        staging_dir=tmp_path / "staging",
        bump="minor",
        reviews_dir=tmp_path / "reviews",
        exporter=_exporter,
        publisher=_publisher,
        open_pr=False,
    )

    assert report.errors == [], report.errors
    assert report.pushed is True
    assert report.tag_pushed is True
    assert (
        _git("branch", "--show-current", cwd=catalog_checkout).stdout.strip() == "main"
    )
    assert _git("status", "--porcelain", cwd=catalog_checkout).stdout == ""
    assert _git("rev-parse", "main", cwd=catalog_checkout).stdout.strip() == main_head
    assert _git(
        "ls-remote",
        "--heads",
        "origin",
        f"refs/heads/{report.branch}",
        cwd=catalog_checkout,
    ).stdout.strip()
    assert _git(
        "ls-remote",
        "--tags",
        "origin",
        f"refs/tags/{report.rc_version}",
        cwd=catalog_checkout,
    ).stdout.strip()


def test_review_release_refuses_dirty_main_before_branch_switch(
    catalog_checkout: Path, tmp_path: Path
) -> None:
    (catalog_checkout / "README.md").write_text("dirty\n", encoding="utf-8")

    def unexpected_export(**_kwargs: object) -> None:
        raise AssertionError("export must not run for a dirty checkout")

    report = run_review_release(
        catalog_checkout,
        _focus_file(tmp_path),
        "Review catalog batch",
        staging_dir=tmp_path / "staging",
        bump="minor",
        reviews_dir=tmp_path / "reviews",
        exporter=unexpected_export,
        publisher=_publisher,
        open_pr=False,
    )

    assert report.errors == [
        "ISNC working tree has 1 uncommitted change(s). Commit changes first."
    ]
    assert (
        _git("branch", "--show-current", cwd=catalog_checkout).stdout.strip() == "main"
    )
    assert "review/" not in _git("branch", cwd=catalog_checkout).stdout
    assert (
        _git("ls-remote", "--heads", "origin", cwd=catalog_checkout).stdout.count(
            "refs/heads/"
        )
        == 1
    )
    assert _git("ls-remote", "--tags", "origin", cwd=catalog_checkout).stdout == ""
