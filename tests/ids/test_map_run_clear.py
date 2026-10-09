"""The engine clear path uses the MAPS_TO_IMAS delete owner.

``imas map run --clear`` must remove the bindings ``imas map clear`` removes.
The engine's clear branch previously reset source status alone, leaving the
previous ``MAPS_TO_IMAS`` edges in place, so a later pass kept stale bindings.
These tests pin that the engine clear path routes its binding deletion through
``clear_mapping_bindings``, the single owner ``map clear`` also uses.
"""

from __future__ import annotations

import asyncio
from unittest.mock import MagicMock

import imas_codex.ids.workers as workers


def _state(**overrides):
    kwargs = {
        "facility": "jt-60sa",
        "target_ids_list": ["magnetics"],
        "clear": True,
    }
    kwargs.update(overrides)
    return workers.MappingDiscoveryState(**kwargs)


def _patch_engine(monkeypatch, called):
    """Keep the engine off the graph and spy the binding-delete owner."""

    def fake_clear_bindings(facility, ids_name, gc):
        called.append((facility, ids_name))
        return 2

    async def fake_run_discovery_engine(state, worker_specs, **kwargs):
        return None

    monkeypatch.setattr(workers, "clear_mapping_bindings", fake_clear_bindings)
    monkeypatch.setattr(workers, "GraphClient", lambda *a, **k: MagicMock())
    monkeypatch.setattr(workers, "reset_mapping_state", lambda *a, **k: 0)
    monkeypatch.setattr(workers, "run_discovery_engine", fake_run_discovery_engine)


def test_engine_clear_path_calls_clear_mapping_bindings(monkeypatch):
    """The clear branch deletes bindings through the owner for the IDS."""
    called: list[tuple[str, str]] = []
    _patch_engine(monkeypatch, called)

    asyncio.run(workers.run_mapping_engine(_state()))

    assert called == [("jt-60sa", "magnetics")]


def test_engine_clear_path_calls_owner_for_each_ids(monkeypatch):
    called: list[tuple[str, str]] = []
    _patch_engine(monkeypatch, called)

    asyncio.run(
        workers.run_mapping_engine(_state(target_ids_list=["magnetics", "equilibrium"]))
    )

    assert called == [("jt-60sa", "magnetics"), ("jt-60sa", "equilibrium")]


def test_engine_without_clear_leaves_bindings_alone(monkeypatch):
    """The owner is not called when --clear is off."""
    called: list[tuple[str, str]] = []
    _patch_engine(monkeypatch, called)

    asyncio.run(workers.run_mapping_engine(_state(clear=False)))

    assert called == []


def test_clear_mapping_bindings_for_ids_sums_deletes(monkeypatch):
    """The helper opens one client and returns the summed delete count."""
    counts = {"magnetics": 3, "equilibrium": 1}
    seen: list[tuple] = []

    def fake_clear_bindings(facility, ids_name, gc):
        seen.append((facility, ids_name))
        return counts[ids_name]

    monkeypatch.setattr(workers, "clear_mapping_bindings", fake_clear_bindings)
    monkeypatch.setattr(workers, "GraphClient", lambda *a, **k: MagicMock())

    total = workers.clear_mapping_bindings_for_ids(
        "jt-60sa", ["magnetics", "equilibrium"]
    )

    assert total == 4
    assert seen == [("jt-60sa", "magnetics"), ("jt-60sa", "equilibrium")]


def test_clear_mapping_bindings_for_ids_no_ids_is_zero(monkeypatch):
    def boom(*a, **k):
        raise AssertionError("GraphClient must not open for an empty IDS list")

    monkeypatch.setattr(workers, "GraphClient", boom)

    assert workers.clear_mapping_bindings_for_ids("jt-60sa", []) == 0
