"""The deterministic fallback PR body must link the catalog reviewing guide.

A reviewer who reaches the fallback body never sees the catalog document
that states which fields they may edit unless the body carries it, and the
release workflow's posted-body gate asserts the substring ``REVIEWING.md``
(the ``gh api ... .body | contains`` check). The link must be a named
markdown link, never a bare URL, so it stays readable and clickable in the
review column.
"""

import re

from imas_codex.standard_names.release_notes import (
    REVIEWING_GUIDE_URL,
    static_pr_notes,
)

_GUIDE_LINK = re.compile(
    r"\[(?P<text>[^\]]*REVIEWING[^\]]*)\]"
    r"\((?P<url>https?://[^)\s]+/REVIEWING\.md)\)"
)


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


def test_static_body_links_the_catalog_reviewing_guide():
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


def test_reviewing_guide_url_renders_as_a_markdown_target():
    body = _fallback_body()

    # The repository's own posted-body gate greps for REVIEWING.md, so the
    # composed body must carry the file name whatever the presentation.
    assert "REVIEWING.md" in body

    # The guide URL must appear only as a markdown-link target — a bare URL
    # dropped into the prose (no link text, no brackets) fails this assertion.
    assert f"]({REVIEWING_GUIDE_URL})" in body
