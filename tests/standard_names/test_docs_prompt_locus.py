"""The docs prompts name the locus without restating its definition.

A locus is defined once, in the locus registry, and its defining sentence is
appended to every name evaluated there. The writing prompts must therefore tell
the model to name the locus where a quantity is evaluated and never define it,
and must not invite a paraphrased second definition.
"""

from __future__ import annotations

from imas_codex.llm.prompt_loader import render_prompt
from imas_codex.standard_names.context import (
    build_compose_context,
    conventions_context_for,
    locus_context_for,
)

_MEMBER = "flux_surface_averaged_xenon_density_at_plasma_boundary"

_APPENDED_INSTRUCTION = "appended from the registry"
_GLOSS_INSTRUCTION = "do NOT quote verbatim"


def _render(name: str) -> str:
    item = {
        "name": _MEMBER,
        "unit": "1/m^3",
        "kind": "scalar",
        "physics_domain": "transport",
    }
    context = {
        **build_compose_context(),
        "sn_name": _MEMBER,
        "unit": "1/m^3",
        "kind": "scalar",
        "physics_domain": "transport",
        "item": item,
        "chain_history": [],
        "nearby_existing_names": [],
        "compose_scored_examples": [],
        "locus_context": locus_context_for(_MEMBER),
        "conventions": conventions_context_for(_MEMBER),
    }
    return render_prompt(name, context)


def test_generate_docs_prompt_names_locus_without_defining_it() -> None:
    rendered = _render("sn/generate_docs_user")

    assert _APPENDED_INSTRUCTION in rendered
    assert _GLOSS_INSTRUCTION not in rendered


def test_refine_docs_prompt_names_locus_without_defining_it() -> None:
    rendered = _render("sn/refine_docs_user")

    assert _APPENDED_INSTRUCTION in rendered
    assert _GLOSS_INSTRUCTION not in rendered
