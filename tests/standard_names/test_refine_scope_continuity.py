"""Focused refine chains retain scope independently of run accounting."""

from __future__ import annotations

import asyncio
from contextlib import contextmanager
from types import SimpleNamespace
from unittest.mock import MagicMock, patch

import pytest


class _GraphClient:
    def __enter__(self):
        return self

    def __exit__(self, *_args):
        return False

    def query(self, *_args, **_kwargs):
        return []


class _LLMResult:
    input_tokens = 21
    output_tokens = 8
    cache_read_tokens = 0
    cache_creation_tokens = 0

    def __init__(self, name: str) -> None:
        self.parsed = SimpleNamespace(
            name=name,
            description=f"Description of {name}",
            kind="scalar",
            reason="Resolve the review finding",
        )

    def __iter__(self):
        return iter((self.parsed, 0.02, 29))


def _manager(audit_run_id: str) -> MagicMock:
    manager = MagicMock()
    manager.run_id = audit_run_id
    lease = MagicMock()
    lease.reserved = 1.0
    lease.release_unused.return_value = 0.0
    manager.reserve.return_value = lease
    return manager


def _item(name: str, scope_run_id: str | None, chain_length: int = 0) -> dict:
    return {
        "id": name,
        "description": f"Description of {name}",
        "kind": "scalar",
        "unit": "m",
        "physics_domain": "magnetics",
        "source_paths": ["dd:pf_active/coil/element/geometry/outline/r"],
        "claim_token": f"claim-{chain_length}",
        "chain_length": chain_length,
        "chain_history": [],
        "scope_run_id": scope_run_id,
    }


@contextmanager
def _worker_dependencies(candidate_names: list[str], persist_calls: list[dict]):
    candidates = iter(candidate_names)

    async def _call_llm(**_kwargs):
        return _LLMResult(next(candidates))

    def _persist(**kwargs):
        persist_calls.append(kwargs)
        return {"old_name": kwargs["old_name"], "new_name": kwargs["new_name"]}

    with (
        patch("imas_codex.graph.client.GraphClient", return_value=_GraphClient()),
        patch(
            "imas_codex.discovery.base.llm.acall_llm_structured",
            side_effect=_call_llm,
        ),
        patch("imas_codex.llm.prompt_loader.render_prompt", return_value="prompt"),
        patch(
            "imas_codex.standard_names.context.build_compose_context",
            return_value={},
        ),
        patch(
            "imas_codex.standard_names.example_loader.load_compose_examples",
            return_value=[],
        ),
        patch(
            "imas_codex.standard_names.workers._hybrid_search_neighbours",
            return_value=[],
        ),
        patch(
            "imas_codex.standard_names.workers._enrich_dd_path_context",
            return_value=None,
        ),
        patch(
            "imas_codex.standard_names.canonical.find_name_key_duplicate",
            return_value=None,
        ),
        patch(
            "imas_codex.standard_names.fanout.load_settings",
            return_value=SimpleNamespace(
                enabled=False,
                sites={},
                refine_trigger_keywords=(),
                refine_trigger_comment_dims=(),
                refine_trigger_comment_chars=0,
            ),
        ),
        patch("imas_codex.settings.get_model", return_value="test/refiner"),
        patch(
            "imas_codex.standard_names.workers.model_provider_exposure",
            return_value=1.0,
        ),
        patch(
            "imas_codex.standard_names.graph_ops.persist_refined_name",
            side_effect=_persist,
        ),
        patch("imas_codex.standard_names.graph_ops.bump_sn_run_counter") as bump,
    ):
        yield bump


@pytest.mark.asyncio
async def test_focused_scope_survives_multiple_refine_rotations() -> None:
    from imas_codex.standard_names.workers import process_refine_name_batch

    scope = "source-focus-scope"
    audit = "audit-run"
    calls: list[dict] = []
    with _worker_dependencies(["radial_outline", "radial_conductor_outline"], calls):
        manager = _manager(audit)
        assert (
            await process_refine_name_batch(
                [_item("radial_coordinate", scope)],
                manager,
                asyncio.Event(),
                scope_run_id=scope,
            )
            == 1
        )
        successor_scope = calls[-1]["run_id"] or scope
        assert (
            await process_refine_name_batch(
                [_item("radial_outline", successor_scope, chain_length=1)],
                manager,
                asyncio.Event(),
                scope_run_id=scope,
            )
            == 1
        )

    assert [call["run_id"] for call in calls] == [None, None]


@pytest.mark.asyncio
async def test_focused_refine_keeps_audit_counter_separate() -> None:
    from imas_codex.standard_names.workers import process_refine_name_batch

    calls: list[dict] = []
    manager = _manager("audit-run")
    with _worker_dependencies(["radial_outline"], calls) as bump:
        await process_refine_name_batch(
            [_item("radial_coordinate", "source-focus-scope")],
            manager,
            asyncio.Event(),
            scope_run_id="source-focus-scope",
        )

    assert calls[0]["run_id"] is None
    bump.assert_called_once_with("audit-run", "names_regenerated")
    charged_event = manager.reserve.return_value.charge_event.call_args.args[1]
    assert charged_event.phase == "refine_name"


@pytest.mark.asyncio
async def test_unscoped_refine_retains_audit_run_behavior() -> None:
    from imas_codex.standard_names.workers import process_refine_name_batch

    calls: list[dict] = []
    manager = _manager("audit-run")
    with _worker_dependencies(["radial_outline"], calls) as bump:
        await process_refine_name_batch(
            [_item("radial_coordinate", None)], manager, asyncio.Event()
        )

    assert calls[0]["run_id"] == "audit-run"
    bump.assert_not_called()


@pytest.mark.asyncio
async def test_unscoped_refine_ignores_one_historical_scope() -> None:
    from imas_codex.standard_names.workers import process_refine_name_batch

    calls: list[dict] = []
    manager = _manager("audit-run")
    with _worker_dependencies(["radial_outline"], calls) as bump:
        await process_refine_name_batch(
            [_item("radial_coordinate", "historical-run")],
            manager,
            asyncio.Event(),
        )

    assert calls[0]["run_id"] == "audit-run"
    bump.assert_not_called()


@pytest.mark.asyncio
async def test_unscoped_refine_ignores_mixed_historical_scopes() -> None:
    from imas_codex.standard_names.workers import process_refine_name_batch

    calls: list[dict] = []
    manager = _manager("audit-run")
    items = [
        _item("radial_coordinate", "historical-a"),
        _item("vertical_coordinate", "historical-b"),
        _item("toroidal_coordinate", None),
    ]
    with _worker_dependencies(
        ["radial_outline", "vertical_outline", "toroidal_outline"], calls
    ) as bump:
        processed = await process_refine_name_batch(
            items,
            manager,
            asyncio.Event(),
        )

    assert processed == 3
    assert [call["run_id"] for call in calls] == ["audit-run"] * 3
    bump.assert_not_called()


@pytest.mark.asyncio
async def test_pinned_rename_restage_keeps_existing_focused_scope() -> None:
    from imas_codex.standard_names.workers import process_refine_name_batch

    scope = "source-focus-scope"
    item = _item("radial_coordinate", scope)
    item["edit_mode"] = "rename"
    calls: list[dict] = []
    with (
        _worker_dependencies(["unused"], calls),
        patch(
            "imas_codex.standard_names.graph_ops.resubmit_pinned_rename_for_review",
            return_value="resubmitted",
        ) as resubmit,
    ):
        processed = await process_refine_name_batch(
            [item],
            _manager("audit-run"),
            asyncio.Event(),
            scope_run_id=scope,
        )

    assert processed == 0
    assert item["scope_run_id"] == scope
    resubmit.assert_called_once()
    assert calls == []


@pytest.mark.parametrize(
    "scope_metadata",
    [
        "other-scope",
        None,
        "   ",
        pytest.param("missing", id="missing"),
    ],
)
@pytest.mark.asyncio
async def test_inconsistent_focused_metadata_refuses_before_persistence(
    scope_metadata: str | None,
) -> None:
    from imas_codex.standard_names.workers import (
        RefineScopeContinuityError,
        process_refine_name_batch,
    )

    calls: list[dict] = []
    item = _item("radial_coordinate", scope_metadata)
    if scope_metadata == "missing":
        item.pop("scope_run_id")
    with _worker_dependencies(["unused"], calls):
        with pytest.raises(RefineScopeContinuityError):
            await process_refine_name_batch(
                [item],
                _manager("audit-run"),
                asyncio.Event(),
                scope_run_id="source-focus-scope",
            )
    assert calls == []


@pytest.mark.asyncio
async def test_pool_adapter_passes_explicit_refine_scope() -> None:
    from imas_codex.standard_names import loop

    captured: dict = {}

    async def _process(_items, _manager, _stop_event, **kwargs):
        captured.update(kwargs)
        return 1

    with (
        patch(
            "imas_codex.standard_names.workers.process_refine_name_batch",
            side_effect=_process,
        ),
        patch("imas_codex.settings.get_pool_replicas", return_value=1),
    ):
        specs = loop._build_pool_specs(
            _manager("audit-run"),
            asyncio.Event(),
            scope_run_id="source-focus-scope",
            scope_size_hint=1,
            only_pool="refine_name",
        )
        processed = await specs[0].process(
            {"items": [_item("radial_coordinate", "source-focus-scope")]}
        )

    assert processed == 1
    assert captured["scope_run_id"] == "source-focus-scope"


def test_atomic_claim_returns_canonical_scope_identity() -> None:
    from imas_codex.standard_names import graph_ops

    transaction = MagicMock()
    transaction.closed = False
    transaction.run.side_effect = [
        [{"_cluster_id": None, "_unit": "m", "_physics_domain": "magnetics"}],
        [
            {
                "id": "radial_coordinate",
                "claim_token": "claim-token",
                "claim_seq": 1,
                "scope_run_id": "source-focus-scope",
            }
        ],
    ]
    session = MagicMock()
    session.begin_transaction.return_value = transaction
    graph = MagicMock()
    graph.__enter__.return_value = graph
    graph.__exit__.return_value = False

    @contextmanager
    def _session():
        yield session

    graph.session = _session
    with (
        patch("imas_codex.standard_names.graph_ops.GraphClient", return_value=graph),
        patch.object(
            graph_ops,
            "_verify_name_claim_winners",
            side_effect=lambda items, **_kwargs: items,
        ),
        patch(
            "imas_codex.standard_names.chain_history.name_chain_history",
            return_value=[],
        ),
    ):
        claimed = graph_ops.claim_refine_name_batch(batch_size=1)

    readback_query = transaction.run.call_args_list[1].args[0]
    assert "sn.run_id AS scope_run_id" in readback_query
    assert claimed[0]["scope_run_id"] == "source-focus-scope"


@pytest.mark.asyncio
async def test_focused_successor_remains_claimable_by_same_scope() -> None:
    from imas_codex.standard_names.workers import process_refine_name_batch

    scope = "source-focus-scope"
    calls: list[dict] = []
    with _worker_dependencies(["radial_outline"], calls):
        await process_refine_name_batch(
            [_item("radial_coordinate", scope)],
            _manager("audit-run"),
            asyncio.Event(),
            scope_run_id=scope,
        )

    successor_scope = calls[0]["run_id"] or scope
    available = [{"id": "radial_outline", "run_id": successor_scope}]

    claimed = [row for row in available if row["run_id"] == scope]
    assert [row["id"] for row in claimed] == ["radial_outline"]
