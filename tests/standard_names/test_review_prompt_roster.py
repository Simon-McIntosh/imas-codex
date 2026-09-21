"""The catalog roster reaches the rendered review prompt.

The review context supplies the catalog's accepted names through the
``existing_names`` variable. Jinja resolves an undefined variable to nothing
without raising, so a template that never reads the variable discards the
roster silently. A test that asserts on the context dict, or that replaces the
renderer with a stub, cannot observe that: the variable is present in the
context either way. This test renders the real template and asserts the
supplied id is a substring of the returned text.
"""

from __future__ import annotations

import pytest

CATALOG_ROSTER_MARKER = "MARKER_CATALOG_ROSTER_ENTRY"


@pytest.mark.parametrize("prompt_name", ["sn/review_names", "sn/review_docs"])
def test_catalog_roster_reaches_rendered_prompt(prompt_name: str) -> None:
    from imas_codex.llm.prompt_loader import render_prompt

    rendered = render_prompt(
        prompt_name,
        {
            "items": [],
            "existing_names": [CATALOG_ROSTER_MARKER],
        },
    )

    assert CATALOG_ROSTER_MARKER in rendered
