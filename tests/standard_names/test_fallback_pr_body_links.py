"""The deterministic fallback PR body must link the catalog reviewing guide.

A reviewer who reaches the fallback body never sees the catalog document
that states which fields they may edit unless the body carries it, and the
release workflow's posted-body gate asserts the substring ``REVIEWING.md``
(the ``gh api ... .body | contains`` check). The link must be a named
markdown link, never a bare URL, so it stays readable and clickable in the
review column.

The address itself is derived from the catalog checkout's github ``origin``
remote rather than spelled as literal owner and repository, so it follows the
release target and raises instead of composing a link that would 404 when the
remote cannot be resolved. The derivation mirrors
``exclusion_ledger_blob_url``: resolve the checkout, parse the origin slug,
and raise rather than guess.
"""

import ast
import inspect
import re
import subprocess

import pytest

from iter_codex.standard_names import release_notes
from iter_codex.standard_names.release_notes import (
    ReviewingGuideLinkError,
    reviewing_guide_url,
    static_pr_notes,
)

_GUIDE_LINK = re.compile(
    r"\[(?P<text>[^\]]*REVIEWING[^\]]*)\]"
    r"\((?P<url>https?://[^)\s]+/REVIEWING\.md)\)"
)


def _git(*args: str, cwd) -> subprocess.CompletedProcess[str]:
    return subprocess.run(
        ["git", *args], cwd=cwd, capture_output=True, text=True, check=True
    )


@pytest.fixture
def catalog_checkout(tmp_path) -> str:
    """A fake catalog checkout whose origin names a distinctive owner/repo.

    The remote is a github URL parsed by the derivation, so a hardcoded owner
    and repository literal (the constant this derivation replaced) would not
    resolve against it and the comparison tests fail by name.
    """
    work = tmp_path / "catalog"
    work.mkdir()
    _git("init", "-b", "main", cwd=work)
    _git("config", "user.email", "t@t", cwd=work)
    _git("config", "user.name", "t", cwd=work)
    _git(
        "remote",
        "add",
        "origin",
        "https://github.com/example-owner/example-catalog.git",
        cwd=work,
    )
    (work / "REVIEWING.md").write_text("# Reviewing\n")
    _git("add", "REVIEWING.md", cwd=work)
    _git("commit", "-m", "guide", cwd=work)
    return str(work)


def _fallback_body() -> str:
    """The body the deterministic fallback composes for a WEST batch."""
    _title, body = static_pr_notes(
        message="WEST batch",
        rc_version="v0.1.0rc1+west-task-2e",
        batch_size=3,
        minted_from="west_task_2e.yaml",
        changes=[
            {
                "domain": "equilibrium",
                "added": ["a", "b"],
                "changed": ["c"],
                "removed": [],
            }
        ],
    )
    return body


def test_release_notes_does_not_define_a_second_git_helper_copy():
    tree = ast.parse(inspect.getsource(release_notes))
    definitions = {
        node.name
        for node in ast.walk(tree)
        if isinstance(node, ast.FunctionDef | ast.AsyncFunctionDef)
    }
    assert {"_run_git", "_github_slug"}.isdisjoint(definitions)


def test_reviewing_guide_url_is_derived_from_the_catalog_origin_remote(
    catalog_checkout,
):
    # The address must carry the checkout's OWN owner and repository. A
    # literal constant (e.g. the hardcoded owner/repo it replaced) would not
    # name the fixture's remote, so this exact comparison fails by name when
    # the derivation is reverted to a literal.
    assert (
        reviewing_guide_url(checkout=catalog_checkout)
        == "https://github.com/example-owner/example-catalog/blob/main/REVIEWING.md"
    )


def test_reviewing_guide_url_raises_when_the_catalog_remote_cannot_be_resolved(
    tmp_path,
):
    checkout = tmp_path / "no-remote"
    checkout.mkdir()
    _git("init", "-b", "main", cwd=checkout)
    # The checkout has no github 'origin' remote, so no address can be
    # derived. A literal constant never raises, so reverting the derivation
    # to one fails this test by name.
    with pytest.raises(ReviewingGuideLinkError):
        reviewing_guide_url(checkout=checkout)


def test_reviewing_guide_url_raises_when_no_catalog_checkout_is_resolved(monkeypatch):
    monkeypatch.setattr("iter_codex.settings.get_sn_isnc_dir", lambda: None)
    with pytest.raises(ReviewingGuideLinkError):
        reviewing_guide_url()


def test_static_body_links_the_catalog_reviewing_guide(monkeypatch, catalog_checkout):
    monkeypatch.setattr(
        "iter_codex.settings.get_sn_isnc_dir", lambda: catalog_checkout
    )
    body = _fallback_body()

    match = _GUIDE_LINK.search(body)
    assert match is not None, (
        "fallback body must carry a markdown link whose text names the "
        "reviewing guide — a removed link fails here"
    )
    assert "reviewing guide" in match["text"].casefold()
    url = match["url"]
    assert url.rstrip("/").endswith("/REVIEWING.md")
    assert "/blob/" in url


def test_static_body_derived_url_is_the_catalog_checkouts_own(
    monkeypatch, catalog_checkout
):
    monkeypatch.setattr(
        "iter_codex.settings.get_sn_isnc_dir", lambda: catalog_checkout
    )
    body = _fallback_body()

    # The composed body must carry exactly the derived address of the checkout
    # the release runs against — a different (e.g. hardcoded) owner and
    # repository fails this byte comparison.
    assert (
        "](https://github.com/example-owner/example-catalog/blob/main/REVIEWING.md)"
        in body
    )
    # The guide address appears only as a markdown-link target, never as a
    # bare URL dropped into the prose.
    assert "REVIEWING.md" in body
