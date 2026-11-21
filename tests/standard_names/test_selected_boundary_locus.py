"""A boundary the source lets be selected is not the last closed flux surface.

The locus-token binding forced every shape parameter onto ``plasma_boundary``,
whose registry definition is the outermost closed flux surface. Eleven
equilibrium members are sourced from a structure that states the boundary is
*chosen* — ``equilibrium/time_slice/boundary``, "typically chosen at psi_norm =
99.x% of the separatrix" — so their definitions stated a boundary their source
lets be selected.

The repair is a naming rule in the composition system prompt: ``plasma_boundary``
names the last closed flux surface, a selectable boundary is the *effective*
plasma boundary, and because the grammar carries no token for it yet the stage
must emit a ``vocab_gap`` instead of inventing the binding. These tests render
the system prompt the composer is given and hold the rule in it.
"""

from __future__ import annotations

from iter_codex.llm.prompt_loader import render_prompt
from iter_codex.standard_names.context import build_compose_context


def _system_prompt() -> str:
    return render_prompt("sn/generate_name_system", build_compose_context())


def _boundary_rule(prompt: str) -> str:
    """Return the paragraph that binds the boundary locus token."""
    marker = "`plasma_boundary` is the last closed flux surface"
    start = prompt.find(marker)
    assert start >= 0, "the boundary locus rule is absent from the system prompt"
    end = prompt.find("\n- ", start)
    return prompt[start : end if end >= 0 else len(prompt)]


def test_the_boundary_token_is_the_last_closed_flux_surface() -> None:
    rule = _boundary_rule(_system_prompt())

    assert "last closed flux surface" in rule


def test_a_selectable_boundary_must_not_bind_the_boundary_token() -> None:
    rule = _boundary_rule(_system_prompt())

    # The exact source wording of the eleven members.
    assert "psi_norm = 99.x% of the separatrix" in rule
    # The locus the lead ruling names for that selection, and the demand that
    # the stage surface it rather than reuse the categorical token.
    assert "effective_plasma_boundary" in rule
    assert "vocab_gap" in rule
    assert "never bind such a source to `plasma_boundary`" in rule


def test_the_gap_names_a_class_the_grammar_reports() -> None:
    """A gap filed under a class the grammar does not have is dropped unseen.

    ``StandardNameVocabGap`` validates the class against the installed grammar
    and ``_normalise_gap_segments`` drops an unknown one before it can be reconciled,
    so the gap class must be legal or the refusal disappears.
    """
    from iter_codex.standard_names.segments import reportable_segments

    rule = _boundary_rule(_system_prompt())
    assert 'segment: "geometry"' in rule
    assert "geometry" in reportable_segments()


def test_the_stated_last_closed_surface_keeps_the_boundary_token() -> None:
    """The reservation is one-sided: a source that names the LCFS keeps it."""
    rule = _boundary_rule(_system_prompt())

    assert "names the last closed flux surface" in rule
    assert "keeps `plasma_boundary`" in rule


def test_the_prompt_does_not_call_the_true_boundary_a_separatrix() -> None:
    prompt = _system_prompt()
    marker = '`plasma_boundary` never means "separatrix" as a word'
    start = prompt.find(marker)
    assert start >= 0, "the separatrix wording rule is absent"
    rule = prompt[start : prompt.find("\n- ", start)]
    assert "limited plasma" in rule
