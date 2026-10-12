"""An unparseable name yields no conventions instead of aborting the caller.

``conventions_context_for`` parses a name with the public grammar to select its
registry conventions. A name the grammar cannot parse (a legacy id, or one whose
vocabulary an ISN release has not yet reached this checkout) used to raise out of
the function, so an ``sn review`` documentation batch containing one such name
crashed at the batch-reviewer render site. The guard belongs in the selection
owner, not at each call site.
"""

from __future__ import annotations

import logging
from typing import Any

import pytest

from imas_codex.standard_names.context import conventions_context_for

# A name the grammar cannot parse: it is not a valid lowercase snake_case token,
# so ``select_conventions`` raises the parser's ParseError.
UNPARSEABLE = "Legacy-Thing"

# A name the registry selects several entries for, so a "got conventions" check
# cannot pass vacuously on an empty selection.
PARSEABLE = "flux_surface_averaged_argon_density_at_plasma_boundary"


def test_unparseable_name_returns_none() -> None:
    assert conventions_context_for(UNPARSEABLE) is None


def test_parseable_name_still_selects_conventions() -> None:
    selected = conventions_context_for(PARSEABLE)
    assert selected, "precondition: the registry selects conventions for the name"
    assert selected["concepts"] or selected["notation"] or selected["templates"]


def test_guard_catches_only_the_grammar_parse_error(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """A failure that is not the grammar's parse error must not be swallowed."""
    from imas_standard_names.grammar import terms

    def boom(name: str) -> Any:
        raise RuntimeError("registry unavailable")

    monkeypatch.setattr(terms, "select_conventions", boom)

    with pytest.raises(RuntimeError):
        conventions_context_for(PARSEABLE)


@pytest.mark.asyncio
async def test_review_batch_completes_without_conventions_for_unparseable_name(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """The docs batch reviewer drives an unparseable name to completion."""
    from imas_codex.llm import prompt_loader
    from imas_codex.standard_names.review import pipeline as pl

    calls: list[tuple[str, dict]] = []

    def fake_render_prompt(prompt_name: str, context: dict) -> str:
        calls.append((prompt_name, context))
        return "stub-prompt"

    monkeypatch.setattr(prompt_loader, "render_prompt", fake_render_prompt)

    async def fake_call_llm_structured(**kwargs: Any):
        return kwargs["response_model"](reviews=[]), 0.0, 0

    import imas_codex.discovery.base.llm as dbllm

    monkeypatch.setattr(dbllm, "acall_llm_structured", fake_call_llm_structured)

    wlog = logging.LoggerAdapter(logging.getLogger("test"), {})

    await pl._review_single_batch(
        names=[{"id": UNPARSEABLE, "standard_name": UNPARSEABLE}],
        model="fake/model",
        grammar_enums={},
        compose_ctx={},
        batch_context="",
        neighborhood=[],
        audit_findings=[],
        wlog=wlog,
        target="docs",
    )

    populated = [
        ctx for name, ctx in calls if name == "sn/review_docs" and ctx.get("items")
    ]
    assert populated, "the reviewer never rendered a populated item list"
    assert populated[-1]["items"][0]["conventions"] is None
