"""Bounded input allowance for routes newer than the LiteLLM catalog.

A production seat can precede the bundled LiteLLM catalog, so its route
carries no ``max_input_tokens``. The checked-in pricing table records the
allowance for exactly those routes, and the catalog lookup must fall back to
it; a route recorded by neither must still report no bound so the callers'
bounded-input guards keep refusing an unbounded model.
"""

from __future__ import annotations

import asyncio

import pytest

from imas_codex.discovery.base import llm
from imas_codex.standard_names import workers

LIVE_GPT6_SEATS = (
    "openrouter/openai/gpt-6.1-sol",
    "openrouter/openai/gpt-6-luna",
)


@pytest.mark.parametrize("model", LIVE_GPT6_SEATS)
def test_live_gpt6_seat_resolves_positive_input_allowance(model):
    """Each live GPT-6 seat carries a positive bounded input allowance."""
    bound = llm.get_catalog_model_info(model).get("max_input_tokens")
    assert isinstance(bound, int)
    assert bound > 0


def test_route_without_catalog_or_table_allowance_has_none():
    """A route recorded by neither owner still reports no input bound."""
    info = llm.get_catalog_model_info("openrouter/nonexistent/vendor-model")
    assert info.get("max_input_tokens") is None


def test_docs_review_escalator_accepts_rotated_breaker():
    """The docs-review escalator bound accepts the rotated GPT-6 breaker.

    ``prepare_docs_review_request`` refuses with
    ``escalator catalog input allowance is not bounded`` when the third
    (breaker) model in the docs-review chain resolves no catalog bound. The
    rotated breaker is a GPT-6 seat, so this exercises the same guard the
    harmonize apply hits in production.
    """
    item = {
        "id": "test/rotated_breaker",
        "name": "rotated_breaker",
        "description": "placeholder description",
        "documentation": "placeholder documentation",
        "source_bindings": [],
        "source_paths": [],
    }
    prepared = asyncio.run(workers.prepare_docs_review_request(item))
    breaker = prepared["models"][2]
    assert breaker == "openrouter/openai/gpt-6.1-sol"
    assert prepared["escalation_input_token_bound"] > 0
    assert prepared["conditional_exposure"] > 0
