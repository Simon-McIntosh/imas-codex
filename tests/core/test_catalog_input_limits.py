"""Bounded input allowance for routes newer than the LiteLLM catalog.

A production seat can precede the bundled LiteLLM catalog, so its route
carries no ``max_input_tokens``. The checked-in pricing table records the
allowance for exactly those routes, and the catalog lookup must fall back to
it; a route recorded by neither must still report no bound so the callers'
bounded-input guards keep refusing an unbounded model.
"""

from __future__ import annotations

import asyncio
import tomllib
from pathlib import Path

import pytest

from imas_codex.discovery.base import llm
from imas_codex.settings import get_openrouter_pricing
from imas_codex.standard_names import workers
from imas_codex.standard_names.worker_tasks import (
    worker_review_docs as _owning_worker_review_docs,
)

LIVE_GPT6_SEATS = (
    "openrouter/openai/gpt-6.1-sol",
    "openrouter/openai/gpt-6-luna",
)

_REPO_ROOT = Path(__file__).resolve().parents[2]
_PRICING_TABLE = (
    tomllib.loads((_REPO_ROOT / "pyproject.toml").read_text())
    .get("tool", {})
    .get("imas-codex", {})
    .get("llm", {})
    .get("openrouter-pricing", {})
)
_TABLE_ROUTES = sorted(_PRICING_TABLE)


def test_pricing_owner_exposes_the_recorded_input_allowance():
    """The settings owner returns the table's ``max-input-tokens``."""
    entry = get_openrouter_pricing("openrouter/openai/gpt-6.1-sol")
    assert entry["max_input_tokens"] == 1050000


def test_pricing_owner_reports_no_allowance_for_an_unlisted_route():
    """A route with no table entry reports no input allowance."""
    entry = get_openrouter_pricing("openrouter/nonexistent/vendor-model")
    assert entry.get("max_input_tokens") is None


@pytest.mark.parametrize("route", _TABLE_ROUTES)
def test_catalog_allowance_is_unchanged_for_every_table_route(route):
    """``get_catalog_model_info`` keeps the bound each route resolves today.

    Where the LiteLLM catalog records a bound it stays the source; where it
    records none the table's allowance fills the gap. Either way the value is
    the one the base module produced, so routing the read through the settings
    owner changes behaviour for no route.
    """
    catalog = llm._lookup_catalog(route, route_only=False)[0].get("max_input_tokens")
    expected = (
        catalog if catalog is not None else _PRICING_TABLE[route]["max-input-tokens"]
    )
    assert llm.get_catalog_model_info(route).get("max_input_tokens") == expected


def test_catalog_allowance_resolves_for_the_bare_route_spelling():
    """The unprefixed spelling still resolves the table's allowance.

    The owner is keyed by the ``openrouter/``-prefixed route, so the bare
    vendor spelling must be normalized before the lookup or a bare-spelled
    seat silently loses its bound.
    """
    info = llm.get_catalog_model_info("openai/gpt-6.1-sol")
    assert info.get("max_input_tokens") == 1050000


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
    prepared = asyncio.run(_owning_worker_review_docs.prepare_docs_review_request(item))
    breaker = prepared["models"][2]
    assert breaker == "openrouter/openai/gpt-6.1-sol"
    assert prepared["escalation_input_token_bound"] > 0
    assert prepared["conditional_exposure"] > 0
