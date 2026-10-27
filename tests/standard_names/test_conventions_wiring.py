"""The selected documentation conventions reach each documentation render site.

The prompt-level tests hand a template its conventions directly, so a render
site that stops supplying the ``conventions`` key leaves every prompt test
green while the block silently drops out of the live prompt. These tests drive
each site itself with ``render_prompt`` stubbed and assert the context the site
passes carries the conventions the registry selects for the item.

One test per site: the two documentation-generation workers, the docs-review
request builder, and the batch reviewer.
"""

from __future__ import annotations

import asyncio
import logging
from typing import Any
from unittest.mock import MagicMock

import pytest

from imas_codex.standard_names.context import conventions_context_for
from imas_codex.standard_names.worker_tasks import (
    worker_generate_docs as _owning_worker_generate_docs,
    worker_refine_docs as _owning_worker_refine_docs,
    worker_review_docs as _owning_worker_review_docs,
)

# A name the registry selects several entries for (concepts, notation and a
# documentation template), so the assertion cannot pass vacuously on an empty
# selection.
NAME = "flux_surface_averaged_argon_density_at_plasma_boundary"


class _StubGraph:
    """Minimal GraphClient stand-in: context manager whose query returns nothing."""

    def __enter__(self) -> _StubGraph:
        return self

    def __exit__(self, *exc: object) -> bool:
        return False

    def query(self, cypher: str, **params: Any) -> list[dict[str, Any]]:
        return []


def _capture_render_prompt(monkeypatch: pytest.MonkeyPatch) -> list[tuple[str, dict]]:
    """Patch the prompt loader's render_prompt to record each call's context."""
    from imas_codex.llm import prompt_loader

    calls: list[tuple[str, dict]] = []

    def fake_render_prompt(prompt_name: str, context: dict) -> str:
        calls.append((prompt_name, context))
        return "stub-prompt"

    monkeypatch.setattr(prompt_loader, "render_prompt", fake_render_prompt)
    return calls


def _stub_graph(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(
        "imas_codex.graph.client.GraphClient", lambda *a, **k: _StubGraph()
    )


def _context_for(calls: list[tuple[str, dict]], prompt_name: str) -> dict:
    matches = [ctx for name, ctx in calls if name == prompt_name]
    assert matches, f"{prompt_name} was never rendered"
    return matches[-1]


# ---------------------------------------------------------------------------
# Site 1 — process_generate_docs_batch (workers.py)
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_generate_docs_batch_supplies_conventions(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from imas_codex.standard_names import workers

    selected = conventions_context_for(NAME)
    assert selected, "precondition: the registry selects conventions for the name"

    calls = _capture_render_prompt(monkeypatch)
    _stub_graph(monkeypatch)

    mgr = MagicMock()
    mgr.reserve.return_value = None  # stop the loop before the LLM call
    stop = asyncio.Event()
    item: dict[str, Any] = {
        "id": NAME,
        "claim_token": "",
        "kind": "scalar",
        "unit": "m^-3",
        "chain_history": [],
    }

    await _owning_worker_generate_docs.process_generate_docs_batch([item], mgr, stop)

    ctx = _context_for(calls, "sn/generate_docs_user")
    assert ctx["conventions"] == selected


# ---------------------------------------------------------------------------
# Site 2 — prepare_docs_review_request (workers.py)
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_prepare_docs_review_request_supplies_conventions(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from imas_codex.standard_names import workers

    selected = conventions_context_for(NAME)
    assert selected, "precondition: the registry selects conventions for the name"

    calls = _capture_render_prompt(monkeypatch)
    _stub_graph(monkeypatch)
    monkeypatch.setattr(
        "imas_codex.standard_names.context.fetch_review_neighbours",
        lambda item: {
            "vector_neighbours": [],
            "same_base_neighbours": [],
            "same_path_neighbours": [],
        },
    )

    item: dict[str, Any] = {
        "id": NAME,
        "name": NAME,
        "description": "desc",
        "documentation": "docs",
        "kind": "scalar",
        "unit": "m^-3",
        "physics_domain": ["core_profiles"],
        "docs_stage": "draft",
    }

    await _owning_worker_review_docs.prepare_docs_review_request(item)

    ctx = _context_for(calls, "sn/review_docs_user")
    assert ctx["conventions"] == selected


# ---------------------------------------------------------------------------
# Site 3 — process_refine_docs_batch (workers.py)
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_refine_docs_batch_supplies_conventions(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from imas_codex.standard_names import workers

    selected = conventions_context_for(NAME)
    assert selected, "precondition: the registry selects conventions for the name"

    calls = _capture_render_prompt(monkeypatch)
    _stub_graph(monkeypatch)

    mgr = MagicMock()
    mgr.reserve.return_value = None  # stop the loop before the LLM call
    stop = asyncio.Event()
    item: dict[str, Any] = {
        "id": NAME,
        "description": "desc",
        "documentation": "docs",
        "kind": "scalar",
        "unit": "m^-3",
        "physics_domain": ["core_profiles"],
        "docs_chain_length": 0,
        "docs_stage": "refining",
        "claim_token": "tok",
        "docs_chain_history": [],
    }

    await _owning_worker_refine_docs.process_refine_docs_batch([item], mgr, stop)

    ctx = _context_for(calls, "sn/refine_docs_user")
    assert ctx["conventions"] == selected


# ---------------------------------------------------------------------------
# Site 4 — _review_single_batch (review/pipeline.py)
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_review_single_batch_supplies_conventions(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from imas_codex.standard_names.review import pipeline as pl

    selected = conventions_context_for(NAME)
    assert selected, "precondition: the registry selects conventions for the name"

    calls = _capture_render_prompt(monkeypatch)

    async def fake_call_llm_structured(**kwargs: Any):
        return kwargs["response_model"](reviews=[]), 0.0, 0

    import imas_codex.discovery.base.llm as dbllm

    monkeypatch.setattr(dbllm, "acall_llm_structured", fake_call_llm_structured)

    wlog = logging.LoggerAdapter(logging.getLogger("test"), {})

    await pl._review_single_batch(
        names=[{"id": NAME, "standard_name": NAME}],
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
    assert populated[-1]["items"][0]["conventions"] == selected
