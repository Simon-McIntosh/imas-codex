"""Grammar-vocabulary injection into the prompts that judge names.

The composer sees the full closed token registry (``closed_vocab_full`` +
``operators_full``) so it can decompose correctly. The name **reviewer** must
see the SAME registry: without it, a valid name built on a less-common but
registered base (e.g. ``etendue``, ``opacity``) reads to the reviewer as an
unknown base, biasing the score down so the name never accepts. This pins the
requirement that the active review-name prompt injects the full registry.
"""

from __future__ import annotations

import pytest


@pytest.fixture(scope="module")
def compose_ctx():
    from imas_codex.standard_names.context import build_compose_context

    return build_compose_context()


def test_compose_context_carries_the_full_base_registry(compose_ctx):
    """A rare-but-registered base is present in the shared compose context."""
    bases = [
        vs["tokens"]
        for vs in compose_ctx["closed_vocab_full"]
        if vs.get("segment") == "physical_base"
    ]
    assert bases, "no physical_base segment in closed_vocab_full"
    assert "etendue" in bases[0]
    assert "opacity" in bases[0]


def test_review_names_system_injects_the_full_registry(compose_ctx):
    """The active review-name system prompt renders the full token registry."""
    from imas_codex.llm.prompt_loader import render_prompt

    out = render_prompt("sn/review_names_system", compose_ctx)
    # A registered base the composer can legitimately use must be visible to
    # the reviewer, else the reviewer cannot tell it is a valid token.
    assert "etendue" in out
    # The operator registry likewise (a grammar mechanism the reviewer scores).
    assert "flux_surface_averaged" in out


@pytest.mark.parametrize("seat", ["sn/review_names_system", "sn/refine_name_system"])
def test_active_name_seats_inject_public_advisory_aliases(compose_ctx, seat):
    """Every active name critic/repair seat sees the grammar-owned aliases."""
    from imas_codex.llm.prompt_loader import render_prompt

    aliases = compose_ctx["grammar"]["advisory_aliases"]
    out = render_prompt(seat, compose_ctx)

    assert "{{" not in out
    assert "{%" not in out
    for segment_aliases in aliases.values():
        for alias, details in segment_aliases.items():
            assert f"`{alias}`" in out
            assert f"`{details['canonical']}`" in out
