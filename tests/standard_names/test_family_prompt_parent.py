"""Render-guard tests for the sibling-family prompt templates.

A sibling family whose base quantity names no ``StandardName`` node carries
``parent=None`` instead of a bare base token.  The three sibling-family prompt
templates must render in that case as well as when a parent is present, and the
parentless render must carry no ``name:`` link to a parent.

Each template is rendered through the real prompt loader (``render_prompt``),
so a template that dereferences ``parent.name`` unguarded raises under the
loader's ``StrictUndefined`` and fails here.

No LLM calls are made; no graph is touched.
"""

from __future__ import annotations

from collections.abc import Callable

import pytest

from imas_codex.llm.prompt_loader import render_prompt

_PARENT_NAME = "electron_number_density"
_SIBLING_NAME = "electron_number_density_at_pedestal_top"

# The exact link form the parent cross-reference emits; unique, so it does not
# collide with the sibling name that shares the base as a substring.
_PARENT_LINK = f"(name:{_PARENT_NAME})"


def _family(parent_present: bool) -> dict:
    """A sibling family matching ``fetch_sibling_family``'s return shape."""
    parent = (
        {
            "name": _PARENT_NAME,
            "description": "Electron number density.",
            "docs_accepted": True,
        }
        if parent_present
        else None
    )
    return {
        "parent": parent,
        "anchor": None,
        "siblings": [
            {
                "name": _SIBLING_NAME,
                "operator_kind": "position",
                "axis": "at_pedestal_top",
                "description": "Electron number density at the pedestal top.",
                "documentation_opening": "",
                "docs_stage": "",
            }
        ],
    }


def _generate_ctx(parent_present: bool) -> dict:
    return {
        "item": {
            "name": _SIBLING_NAME,
            "unit": "m^-3",
            "kind": "scalar",
            "physics_domain": "core_profiles",
            "sibling_family": _family(parent_present),
        },
        "chain_history": [],
        "nearby_existing_names": [],
    }


def _review_ctx(parent_present: bool) -> dict:
    return {
        "item": {"id": _SIBLING_NAME, "unit": "m^-3", "kind": "scalar"},
        "sibling_family": _family(parent_present),
        "vector_neighbours": [],
        "same_base_neighbours": [],
        "same_path_neighbours": [],
    }


def _refine_ctx(parent_present: bool) -> dict:
    return {
        "sn_name": _SIBLING_NAME,
        "description": "",
        "documentation": "Some existing docs.",
        "unit": "m^-3",
        "kind": "scalar",
        "physics_domain": "core_profiles",
        "docs_chain_length": 0,
        "docs_chain_history": [],
        "reviewer_score_docs": None,
        "reviewer_comments_per_dim_docs": None,
        "dd_paths": [],
        "sibling_family": _family(parent_present),
    }


_RENDERERS: dict[str, Callable[[bool], str]] = {
    "generate_docs": lambda p: render_prompt("sn/generate_docs_user", _generate_ctx(p)),
    "review_docs": lambda p: render_prompt("sn/review_docs_user", _review_ctx(p)),
    "refine_docs": lambda p: render_prompt("sn/refine_docs_user", _refine_ctx(p)),
}


def _norm(text: str) -> str:
    return " ".join(text.split())


@pytest.mark.parametrize("template", sorted(_RENDERERS))
def test_parentless_family_renders_without_parent_link(template: str) -> None:
    """A family with ``parent=None`` renders and names no parent."""
    rendered = _RENDERERS[template](False)

    assert rendered, f"{template}: parentless render produced nothing"
    assert _PARENT_LINK not in rendered, (
        f"{template}: parentless render must not link a parent"
    )
    assert "shared parent" not in _norm(rendered), (
        f"{template}: parentless render must not claim a shared parent"
    )
    assert "sharing the parent" not in _norm(rendered), (
        f"{template}: parentless render must not claim a shared parent"
    )


@pytest.mark.parametrize("template", sorted(_RENDERERS))
def test_family_with_parent_names_the_parent(template: str) -> None:
    """A family with a parent renders and surfaces the parent."""
    rendered = _RENDERERS[template](True)

    assert rendered, f"{template}: parent set render produced nothing"
    if template == "generate_docs":
        assert _PARENT_LINK in rendered, (
            "generate_docs: parent set render must link the parent"
        )
    else:
        assert f"shared parent `{_PARENT_NAME}`" in _norm(rendered), (
            f"{template}: parent set render must name the parent"
        )
