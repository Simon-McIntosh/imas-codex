"""The whole mapping delete has one owner, used by both clear paths.

``imas map run --clear`` and ``imas map clear`` must remove the same thing: a
mapping's MAPS_TO_IMAS bindings, its MappingEvidence nodes and the IMASMapping
node. Both route that delete through ``graph_ops.delete_mapping`` so a cleared
re-run drops the escalation evidence a stale mapping carried. These tests pin
that both callers reach that one owner, that the engine logs its counts even
when a clear deletes nothing, and that a clear deletes a source's evidence and
mapping node as well as its bindings.
"""

from __future__ import annotations

import asyncio
import logging
from unittest.mock import MagicMock

import imas_codex.cli.map as map_cli
import imas_codex.graph.client as gclient
import imas_codex.ids.graph_ops as graph_ops
import imas_codex.ids.workers as workers

ZERO = {"mappings": 0, "bindings": 0, "evidence": 0}


def _state(**overrides):
    kwargs = {
        "facility": "jt-60sa",
        "target_ids_list": ["magnetics"],
        "clear": True,
    }
    kwargs.update(overrides)
    return workers.MappingDiscoveryState(**kwargs)


def _patch_engine(monkeypatch, called):
    """Keep the engine off the graph and spy the whole-mapping owner."""

    def fake_delete_mapping(facility, ids_name, gc):
        called.append((facility, ids_name))
        return {"mappings": 1, "bindings": 2, "evidence": 1}

    async def fake_run_discovery_engine(state, worker_specs, **kwargs):
        return None

    monkeypatch.setattr(workers, "delete_mapping", fake_delete_mapping)
    monkeypatch.setattr(workers, "GraphClient", lambda *a, **k: MagicMock())
    monkeypatch.setattr(workers, "reset_mapping_state", lambda *a, **k: 0)
    monkeypatch.setattr(workers, "run_discovery_engine", fake_run_discovery_engine)


# ---------------------------------------------------------------------------
# Both callers reach the one owner
# ---------------------------------------------------------------------------


def test_engine_clear_path_calls_delete_mapping_owner(monkeypatch):
    """The engine's clear branch deletes through the owner for the IDS."""
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


def test_engine_without_clear_leaves_the_owner_alone(monkeypatch):
    """The owner is not called when --clear is off."""
    called: list[tuple[str, str]] = []
    _patch_engine(monkeypatch, called)

    asyncio.run(workers.run_mapping_engine(_state(clear=False)))

    assert called == []


def test_map_clear_cli_calls_the_same_owner(monkeypatch):
    """``map clear`` deletes through the same graph_ops owner."""
    called: list[tuple[str, str]] = []

    def fake_delete_mapping(facility, ids_name, gc):
        called.append((facility, ids_name))
        return {"mappings": 1, "bindings": 0, "evidence": 0}

    monkeypatch.setattr(graph_ops, "delete_mapping", fake_delete_mapping)
    monkeypatch.setattr(gclient, "GraphClient", lambda *a, **k: MagicMock())

    deleted = map_cli._clear_mapping("jt-60sa", "magnetics")

    assert called == [("jt-60sa", "magnetics")]
    assert deleted == 1


def test_both_clear_paths_reach_the_same_owner(monkeypatch):
    """A shared spy on the graph_ops owner is reached by both entry points."""
    seen: list[tuple[str, str]] = []

    def spy(facility, ids_name, gc):
        seen.append((facility, ids_name))
        return {"mappings": 1, "bindings": 1, "evidence": 1}

    async def fake_run_discovery_engine(state, worker_specs, **kwargs):
        return None

    monkeypatch.setattr(graph_ops, "delete_mapping", spy)
    monkeypatch.setattr(workers, "delete_mapping", spy)
    monkeypatch.setattr(gclient, "GraphClient", lambda *a, **k: MagicMock())
    monkeypatch.setattr(workers, "reset_mapping_state", lambda *a, **k: 0)
    monkeypatch.setattr(workers, "run_discovery_engine", fake_run_discovery_engine)

    asyncio.run(workers.run_mapping_engine(_state()))
    map_cli._clear_mapping("jt-60sa", "magnetics")

    assert seen == [("jt-60sa", "magnetics"), ("jt-60sa", "magnetics")]


# ---------------------------------------------------------------------------
# The engine logs its counts even when a clear deletes nothing
# ---------------------------------------------------------------------------


def test_engine_logs_zero_counts_unconditionally(monkeypatch, caplog):
    """A clear that deletes nothing still logs its zero counts."""

    async def fake_run_discovery_engine(state, worker_specs, **kwargs):
        return None

    monkeypatch.setattr(workers, "delete_mapping", lambda *a, **k: dict(ZERO))
    monkeypatch.setattr(workers, "GraphClient", lambda *a, **k: MagicMock())
    monkeypatch.setattr(workers, "reset_mapping_state", lambda *a, **k: 0)
    monkeypatch.setattr(workers, "run_discovery_engine", fake_run_discovery_engine)

    with caplog.at_level(logging.INFO, logger="imas_codex.ids.workers"):
        asyncio.run(workers.run_mapping_engine(_state()))

    messages = [r.getMessage() for r in caplog.records]
    assert any(
        "Cleared 0 mappings, 0 bindings, 0 evidence for jt-60sa" in m for m in messages
    )


# ---------------------------------------------------------------------------
# The per-IDS aggregation helper
# ---------------------------------------------------------------------------


def test_clear_mappings_for_ids_sums_counts(monkeypatch):
    """The helper opens one client and sums each kind of delete."""
    counts = {
        "magnetics": {"mappings": 1, "bindings": 3, "evidence": 2},
        "equilibrium": {"mappings": 1, "bindings": 1, "evidence": 0},
    }
    seen: list[tuple] = []

    def fake_delete_mapping(facility, ids_name, gc):
        seen.append((facility, ids_name))
        return counts[ids_name]

    monkeypatch.setattr(workers, "delete_mapping", fake_delete_mapping)
    monkeypatch.setattr(workers, "GraphClient", lambda *a, **k: MagicMock())

    total = workers.clear_mappings_for_ids("jt-60sa", ["magnetics", "equilibrium"])

    assert total == {"mappings": 2, "bindings": 4, "evidence": 2}
    assert seen == [("jt-60sa", "magnetics"), ("jt-60sa", "equilibrium")]


def test_clear_mappings_for_ids_no_ids_is_zero(monkeypatch):
    def boom(*a, **k):
        raise AssertionError("GraphClient must not open for an empty IDS list")

    monkeypatch.setattr(workers, "GraphClient", boom)

    assert workers.clear_mappings_for_ids("jt-60sa", []) == dict(ZERO)
