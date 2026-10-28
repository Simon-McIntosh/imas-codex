"""Documentation prompts render only conventions selected for their item."""

from __future__ import annotations

import pytest

from imas_codex.llm.prompt_loader import render_prompt
from imas_codex.standard_names.context import conventions_context_for

NAME = "flux_surface_averaged_argon_density_at_plasma_boundary"
UNSELECTED = "vacuum_toroidal_field_strength"
EQUATION = (
    r"\langle f\rangle = \oint f\,dS/|\nabla\psi| "
    r"\,/\, \oint dS/|\nabla\psi|"
)
PROMPTS = (
    "sn/generate_docs_user",
    "sn/refine_docs_user",
    "sn/review_docs_user",
    "sn/review_docs",
)


def _render(name: str, prompt: str) -> str:
    conventions = conventions_context_for(name)
    item = {
        "id": name,
        "name": name,
        "standard_name": name,
        "conventions": conventions,
    }
    return render_prompt(
        prompt,
        {
            "item": item,
            "items": [item],
            "sn_name": name,
            "conventions": conventions,
            "docs_chain_length": 0,
            "docs_chain_history": [],
        },
    )


@pytest.mark.parametrize("prompt", PROMPTS)
def test_documentation_prompt_uses_only_selected_conventions(prompt: str) -> None:
    rendered = _render(NAME, prompt)

    assert rendered.count("## Selected documentation conventions for this name") == 1
    assert EQUATION in rendered
    assert rendered.count("### Concept:") == 2
    assert rendered.count("### Notation:") == 1
    assert rendered.count("### Documentation opening:") == 1
    assert "### Concept: volume average" not in rendered
    assert "### Notation: temperature" not in rendered


@pytest.mark.parametrize("prompt", PROMPTS)
def test_documentation_prompt_has_no_block_without_selected_entry(prompt: str) -> None:
    # Every parseable name now selects at least its notation entry, so the
    # empty selection is reached through the grammar's parse refusal.
    assert conventions_context_for(UNSELECTED) is None
    rendered = _render(UNSELECTED, prompt)

    assert "## Selected documentation conventions for this name" not in rendered

    assert "## Selected documentation conventions for this name" not in rendered


def test_batch_review_keeps_conventions_with_their_candidate() -> None:
    selected = conventions_context_for(NAME)
    rendered = render_prompt(
        "sn/review_docs",
        {
            "items": [
                {"id": NAME, "standard_name": NAME, "conventions": selected},
                {
                    "id": "argon_temperature",
                    "standard_name": "argon_temperature",
                    "conventions": None,
                },
            ]
        },
    )

    assert rendered.count("## Selected documentation conventions for this name") == 1
    assert rendered.index(EQUATION) < rendered.index("### Candidate 2")
