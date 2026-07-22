"""The full operator registry is injected into the compose grammar reference.

Operators are a grammar mechanism separate from ``SEGMENT_TOKEN_MAP``, so they
never appear in ``closed_vocab_full``.  Without the complete list the composer
sees only a hand-written prose subset and mis-slots real operators
(``line_integrated``, ``square``, ``inverse``, ``change_in``, ``variation`` …)
as qualifiers or false vocab gaps — the WEST mop-up plateau root cause.  These
tests pin:

* the context builder groups every ISN operator by attachment kind, derived
  from ISN at runtime (no drift, no hardcoding);
* the shared grammar reference renders the full registry; and
* the line-of-sight guidance names the coordinate *of* the ``line_of_sight``
  locus rather than the (unregistered) ``line_of_sight`` base.
"""

from __future__ import annotations

import pytest

from imas_codex.llm.prompt_loader import PROMPTS_DIR, _get_jinja_env, render_prompt
from imas_codex.standard_names.context import (
    _load_operators_full,
    build_compose_context,
    clear_context_cache,
)


@pytest.fixture(autouse=True)
def _fresh_context():
    clear_context_cache()
    _load_operators_full.cache_clear()
    yield
    clear_context_cache()
    _load_operators_full.cache_clear()


def _isn_operators() -> dict[str, str]:
    """token -> kind, straight from the ISN public grammar context."""
    from imas_standard_names.grammar.context import get_grammar_context

    ops = get_grammar_context()["grammar"]["vocabularies"]["operators"]
    return {t: (spec or {}).get("kind", "unary_prefix") for t, spec in ops.items()}


def test_operators_full_covers_every_isn_operator_without_drift():
    grouped = _load_operators_full()
    assert grouped is not None
    rendered = {e["token"] for toks in grouped.values() for e in toks}
    assert rendered == set(_isn_operators()), (
        "operators_full must mirror the ISN operator vocabulary exactly — "
        "codex never hardcodes or drops operator tokens"
    )
    # Kind grouping matches ISN.
    isn = _isn_operators()
    for kind, entries in grouped.items():
        for e in entries:
            assert isn[e["token"]] == kind


@pytest.mark.parametrize(
    "token,kind",
    [
        ("line_integrated", "unary_prefix"),
        ("line_averaged", "unary_prefix"),
        ("flux_surface_averaged", "unary_prefix"),
        ("square", "unary_prefix"),
        ("inverse", "unary_prefix"),
        ("change_in", "unary_prefix"),
        ("variation", "unary_prefix"),
        ("magnitude", "unary_postfix"),
        ("ratio", "binary"),
    ],
)
def test_previously_misslotted_operators_are_present(token, kind):
    grouped = _load_operators_full()
    assert token in {e["token"] for e in grouped[kind]}


def test_context_builder_exposes_operators_full():
    ctx = build_compose_context()
    assert ctx.get("operators_full"), "operators_full must be in compose context"


def _render_grammar_reference() -> str:
    env = _get_jinja_env(PROMPTS_DIR)
    tmpl = env.get_template("sn/_grammar_reference.md")
    ctx = build_compose_context()
    return tmpl.render(**ctx)


def test_grammar_reference_renders_full_operator_registry():
    text = _render_grammar_reference()
    # Operators the old static prose omitted.
    for tok in ("flux_surface_averaged", "line_integrated", "square", "inverse"):
        assert tok in text, f"{tok} missing from rendered grammar reference"
    assert "operator_kind" in text


def test_compose_prompt_names_line_of_sight_via_locus_not_base():
    text = render_prompt("sn/generate_name_system", build_compose_context())
    assert "radial_coordinate_of_line_of_sight" in text
    # The stale collapsed base form (`radial_line_of_sight`) must be gone.
    assert "radial_line_of_sight" not in text
    # line_of_sight is taught as a locus token, not a base.
    assert 'locus_token="line_of_sight"' in text
