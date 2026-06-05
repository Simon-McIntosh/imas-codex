"""Phase 1.5 dup-guard wiring in process_refine_name_batch (plan 39 §5.2).

Verifies that ``find_name_key_duplicate`` is consulted AFTER B12's final
candidate, BEFORE persisting via ``persist_refined_name``: on a hit the
candidate is dropped, the source is marked ``duplicate_of=<id>`` via the
``on_event`` payload, and ``persist_refined_name`` is *not* called.
"""

from __future__ import annotations

import asyncio
from typing import Any
from unittest.mock import MagicMock

import pytest

from imas_codex.standard_names.budget import BudgetManager, LLMCostEvent
from imas_codex.standard_names.fanout.config import FanoutSettings
from imas_codex.standard_names.workers import process_refine_name_batch


class _NullGraphClient:
    """Stub ``GraphClient`` returning empty results — never reached because
    the dup-guard is patched at the function level."""

    def __enter__(self):
        return self

    def __exit__(self, *a):
        return False

    def query(self, *a, **kw):
        return []


def _item(name: str, *, chain_length: int = 1) -> dict[str, Any]:
    return {
        "id": name,
        "chain_length": chain_length,
        "chain_history": [
            {
                "name": "old_name",
                "model": "test/model",
                "reviewer_score": 0.5,
                "reviewer_verdict": "revise",
                "reviewer_comments_per_dim": {"clarity": "score below threshold"},
            }
        ],
        "source_paths": ["core_profiles/electrons/temperature"],
        "description": "T_e",
        "unit": "eV",
        "physics_domain": "kinetics",
    }


class _RefinedName:
    def __init__(self, name: str) -> None:
        self.name = name
        self.description = "refined description"
        self.kind = "scalar"
        self.reason = "fixes ambiguity"


class _FakeLLMResult:
    def __init__(
        self,
        parsed: _RefinedName,
        *,
        cost: float = 0.001,
        input_tokens: int = 10,
        output_tokens: int = 5,
        cache_read_tokens: int = 3,
        cache_creation_tokens: int = 2,
    ) -> None:
        self.parsed = parsed
        self.cost = cost
        self.input_tokens = input_tokens
        self.output_tokens = output_tokens
        self.cache_read_tokens = cache_read_tokens
        self.cache_creation_tokens = cache_creation_tokens

    def __iter__(self):
        return iter((self.parsed, self.cost, 0))


@pytest.fixture
def patched(monkeypatch: pytest.MonkeyPatch):
    """Patch GraphClient + LLM + persist + claim release."""
    monkeypatch.setattr(
        "imas_codex.graph.client.GraphClient", lambda *a, **kw: _NullGraphClient()
    )

    # Hybrid neighbours search returns nothing.
    monkeypatch.setattr(
        "imas_codex.standard_names.workers._hybrid_search_neighbours",
        lambda gc, path: [],
    )

    # LLM returns a fixed refined name.
    async def _fake_llm(**kwargs):
        return _FakeLLMResult(_RefinedName("electron_temperature"))

    monkeypatch.setattr("imas_codex.discovery.base.llm.acall_llm_structured", _fake_llm)

    # ``acall_llm_structured`` in workers is imported locally.  Patch
    # that namespace too because the worker uses ``from … import …``
    # at the top of the function body.
    # NOTE: it's imported inside the function, so monkeypatching
    # ``imas_codex.discovery.base.llm`` is sufficient.

    persist_calls: list[dict[str, Any]] = []
    release_calls: list[dict[str, Any]] = []
    charge_events: list[tuple[float, LLMCostEvent]] = []

    def _persist(**kwargs):
        persist_calls.append(kwargs)
        return {}

    def _release(*, sn_ids, token):
        release_calls.append({"sn_ids": list(sn_ids), "token": token})

    monkeypatch.setattr(
        "imas_codex.standard_names.graph_ops.persist_refined_name", _persist
    )
    monkeypatch.setattr(
        "imas_codex.standard_names.graph_ops.release_refine_name_failed_claims",
        _release,
    )

    def _charge_event(self, cost, event):
        charge_events.append((cost, event))
        return None

    monkeypatch.setattr(
        "imas_codex.standard_names.budget.BudgetLease.charge_event",
        _charge_event,
    )
    return persist_calls, release_calls, charge_events


async def test_dup_guard_drops_candidate(
    monkeypatch: pytest.MonkeyPatch, patched
) -> None:
    """When ``find_name_key_duplicate`` reports a hit, the worker must
    NOT call ``persist_refined_name`` and must surface ``dup_prevented``
    via ``on_event``."""
    persist_calls, release_calls, _ = patched

    # Dup guard reports an existing duplicate.
    monkeypatch.setattr(
        "imas_codex.standard_names.canonical.find_name_key_duplicate",
        lambda gc, name, *, exclude=None: "Electron_Temperature",
    )

    events: list[dict[str, Any]] = []
    mgr = BudgetManager(total_budget=100.0)
    stop = asyncio.Event()
    processed = await process_refine_name_batch(
        [_item("old_name")],
        mgr=mgr,
        stop_event=stop,
        on_event=lambda e: events.append(e),
    )
    # Candidate dropped → not counted as processed.
    assert processed == 0
    # No persist call — dup_prevented short-circuited the path.
    assert persist_calls == []
    # Claim released back to 'reviewed'.
    assert any(c["sn_ids"] == ["old_name"] for c in release_calls)
    # Outcome surfaced in on_event payload.
    dup_events = [e for e in events if e.get("outcome") == "dup_prevented"]
    assert len(dup_events) == 1
    assert dup_events[0]["duplicate_of"] == "Electron_Temperature"


async def test_dup_guard_miss_proceeds_to_persist(
    monkeypatch: pytest.MonkeyPatch, patched
) -> None:
    """When the dup guard returns ``None`` the persist path is taken."""
    persist_calls, _, _ = patched

    monkeypatch.setattr(
        "imas_codex.standard_names.canonical.find_name_key_duplicate",
        lambda gc, name, *, exclude=None: None,
    )

    mgr = BudgetManager(total_budget=100.0)
    stop = asyncio.Event()
    processed = await process_refine_name_batch(
        [_item("old_name")],
        mgr=mgr,
        stop_event=stop,
    )
    assert processed == 1
    assert len(persist_calls) == 1
    assert persist_calls[0]["new_name"] == "electron_temperature"
    assert persist_calls[0]["old_name"] == "old_name"


async def test_dup_guard_failure_proceeds_to_persist(
    monkeypatch: pytest.MonkeyPatch, patched
) -> None:
    """Dup-guard exceptions must not block the refine path (best-effort)."""
    persist_calls, _, _ = patched

    def _boom(*a, **kw):
        raise RuntimeError("graph unavailable")

    monkeypatch.setattr(
        "imas_codex.standard_names.canonical.find_name_key_duplicate", _boom
    )

    mgr = BudgetManager(total_budget=100.0)
    stop = asyncio.Event()
    processed = await process_refine_name_batch(
        [_item("old_name")],
        mgr=mgr,
        stop_event=stop,
    )
    assert processed == 1
    assert len(persist_calls) == 1


async def test_refine_worker_records_fanout_telemetry(
    monkeypatch: pytest.MonkeyPatch, patched
) -> None:
    """Fanout telemetry records the actual arm and LLM token/cache splits."""
    persist_calls, _, charge_events = patched

    monkeypatch.setattr(
        "imas_codex.standard_names.canonical.find_name_key_duplicate",
        lambda gc, name, *, exclude=None: None,
    )
    monkeypatch.setattr(
        "imas_codex.standard_names.fanout.load_settings",
        lambda: FanoutSettings(enabled=True, sites={"refine_name": True}),
    )
    monkeypatch.setattr(
        "imas_codex.standard_names.fanout.should_trigger_fanout",
        lambda **kwargs: (True, "score below threshold"),
    )
    monkeypatch.setattr(
        "imas_codex.standard_names.fanout.assign_arm",
        lambda *args, **kwargs: "off",
    )

    async def _fake_run_fanout(**kwargs):
        return ""

    monkeypatch.setattr("imas_codex.standard_names.fanout.run_fanout", _fake_run_fanout)

    events: list[dict[str, Any]] = []
    mgr = BudgetManager(total_budget=100.0, run_id="test-run")
    stop = asyncio.Event()
    processed = await process_refine_name_batch(
        [_item("old_name")],
        mgr=mgr,
        stop_event=stop,
        on_event=lambda e: events.append(e),
    )

    assert processed == 1
    assert len(persist_calls) == 1
    assert len(charge_events) == 1

    charged_cost, cost_event = charge_events[0]
    assert charged_cost == pytest.approx(0.001)
    assert cost_event.phase == "refine_name+fanout"
    assert cost_event.batch_id is not None
    assert cost_event.tokens_in == 10
    assert cost_event.tokens_out == 5
    assert cost_event.tokens_cached_read == 3
    assert cost_event.tokens_cached_write == 2

    assert events[-1]["fanout_run_id"] == cost_event.batch_id
    assert events[-1]["fanout_arm"] == "off"
    assert events[-1]["llm_tokens_in"] == 10
    assert events[-1]["llm_tokens_out"] == 5
    assert events[-1]["llm_tokens_cached_read"] == 3
    assert events[-1]["llm_tokens_cached_write"] == 2
