"""The review body carries the exclusion ledger address at most once.

An authored body may already name the ledger by its repository-relative path —
in prose, or as a hand-written markdown link. Appending the blob address in
that case duplicates the same information in two shapes. The appender must
recognise that the ledger is already named and leave the body byte-identical,
and when it does append it must add a markdown link with descriptive text
rather than a second bare address.
"""

from __future__ import annotations

import json
import subprocess
from pathlib import Path

import pytest

from iter_codex.standard_names.catalog_release import body_with_exclusion_ledger_link

_RELATIVE = "iter_codex/standard_names/manifests/batch_dd_paths.exclusions.json"


def _git(*args: str, cwd: Path) -> subprocess.CompletedProcess[str]:
    return subprocess.run(
        ["git", *args], cwd=cwd, capture_output=True, text=True, check=True
    )


@pytest.fixture
def manifest_checkout(tmp_path: Path) -> tuple[Path, Path]:
    """A committed manifest plus its exclusion ledger, with a github origin."""
    root = tmp_path / "codex"
    manifests = root / "iter_codex" / "standard_names" / "manifests"
    manifests.mkdir(parents=True)
    _git("init", "-q", "-b", "main", cwd=root)
    _git("config", "user.email", "release@example.invalid", cwd=root)
    _git("config", "user.name", "Release", cwd=root)
    _git(
        "remote",
        "add",
        "origin",
        "git@github.com:test-owner/iter-codex.git",
        cwd=root,
    )
    manifest = manifests / "batch_dd_paths.yaml"
    manifest.write_text("sources:\n  - magnetics/flux_loop/flux\n", encoding="utf-8")
    ledger = manifests / "batch_dd_paths.exclusions.json"
    ledger.write_text(
        json.dumps(
            {
                "excluded_ineligible": [
                    {
                        "path": "magnetics/flux_loop/name",
                        "reason": "excluded_metadata",
                        "category": "metadata",
                    }
                ]
            }
        ),
        encoding="utf-8",
    )
    _git(
        "add",
        "iter_codex/standard_names/manifests/batch_dd_paths.yaml",
        _RELATIVE,
        cwd=root,
    )
    _git("commit", "-q", "-m", "add batch manifest and ledger", cwd=root)
    return root, manifest


def test_body_already_naming_the_ledger_is_returned_unchanged(
    manifest_checkout: tuple[Path, Path],
) -> None:
    _root, manifest = manifest_checkout
    authored = (
        "Review candidate.\n\n"
        f"Excluded source paths are recorded in [the exclusion ledger]({_RELATIVE})."
    )

    body = body_with_exclusion_ledger_link(authored, manifest)

    assert body == authored


def test_body_lacking_the_ledger_gains_exactly_one_markdown_link(
    manifest_checkout: tuple[Path, Path],
) -> None:
    root, manifest = manifest_checkout
    sha = _git("log", "-1", "--format=%H", "--", _RELATIVE, cwd=root).stdout.strip()
    expected_url = f"https://github.com/test-owner/iter-codex/blob/{sha}/{_RELATIVE}"
    expected_link = (
        "[ledger of excluded source paths and their withholding data dictionary "
        f"node categories]({expected_url})"
    )

    body = body_with_exclusion_ledger_link("Review candidate.", manifest)

    assert body.count(expected_link) == 1
    # The address appears once, wrapped as a named link — never a bare second
    # copy of the URL floating unwrapped in prose.
    assert body.count(expected_url) == 1
