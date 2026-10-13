"""Cover the classifier role benchmark's chat default and its tree option.

The benchmark seats the domain classifier, and two entry points sit below it.
The default routes each batch to the free-text chat classifier; ``use_tree``
routes it to the two-level category-then-domain Choice, which re-judges its
low-confidence paths with the same chat classifier. Both call sites are
replaced with fakes here, so a run reaches the routed function without a paid
call, and the tests fail if the option is dropped or the routing is swapped.
"""

from __future__ import annotations

import asyncio

import pytest

import imas_codex.graph.dd_domain_classifier as ddc
from imas_codex.standard_names.benchmark_roles import run_classifier_bench

GOLD = [
    {
        "path": "core_profiles/profiles_1d/electrons/temperature",
        "expected_domain": "core_plasma_physics",
    },
    {
        "path": "equilibrium/time_slice/global_quantities/ip",
        "expected_domain": "equilibrium",
    },
]
EXPECTED = {row["path"]: row["expected_domain"] for row in GOLD}


def _context(gc, path_ids):
    """Context rows for the named paths, without a graph call."""
    return [
        {
            "id": pid,
            "description": "",
            "documentation": "",
            "units": "",
            "parent_path": "",
            "parent_description": "",
            "siblings": [],
            "ids_name": "",
            "node_category": "",
        }
        for pid in path_ids
    ]


def test_chat_is_the_default_classifier(monkeypatch):
    calls = {"chat": 0, "tree": 0}

    async def fake_chat(batch, *, model, service, valid_domains):
        calls["chat"] += 1
        return [
            {"id": p["id"], "physics_domain": EXPECTED[p["id"]]} for p in batch
        ], 0.0

    async def fake_tree(batch, **kwargs):
        calls["tree"] += 1
        return [], 0.0

    monkeypatch.setattr(ddc, "gather_classification_context", _context)
    monkeypatch.setattr(ddc, "_classify_batch", fake_chat)
    monkeypatch.setattr(ddc, "_classify_batch_tree", fake_tree)

    report = asyncio.run(
        run_classifier_bench(["openrouter/openai/gpt-5.5"], GOLD, gc=object())
    )

    assert calls["chat"] >= 1
    assert calls["tree"] == 0
    assert report.results[0].metrics["accuracy"] == pytest.approx(1.0)


def test_tree_option_routes_to_the_two_level_choice(monkeypatch):
    calls = {"chat": 0, "tree": 0}

    async def fake_chat(batch, *, model, service, valid_domains):
        calls["chat"] += 1
        return [], 0.0

    async def fake_tree(batch, **kwargs):
        calls["tree"] += 1
        return [{"id": p["id"], "physics_domain": "general"} for p in batch], 0.0

    monkeypatch.setattr(ddc, "gather_classification_context", _context)
    monkeypatch.setattr(ddc, "_classify_batch", fake_chat)
    monkeypatch.setattr(ddc, "_classify_batch_tree", fake_tree)

    report = asyncio.run(
        run_classifier_bench(["typesafe/jev-1.13"], GOLD, gc=object(), use_tree=True)
    )

    assert calls["tree"] >= 1
    assert calls["chat"] == 0
    # every path answered "general", so none matches its gold label
    assert report.results[0].metrics["accuracy"] == pytest.approx(0.0)
