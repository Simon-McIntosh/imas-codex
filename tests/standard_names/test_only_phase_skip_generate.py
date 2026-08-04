"""``sn run --only <phase>`` must actually skip the generate phase.

``--only link`` (and any ``--only`` selection that excludes the generate
phase) sets ``skip_generate=True``.  That flag must reach the pool
orchestrator: the ``generate_name`` pool is dropped and the domain
auto-seed sweep is skipped — otherwise ``--only link`` silently composes
new names and seeds new sources instead of running link resolution only.

All graph interaction is mocked (no live Neo4j).
"""

from __future__ import annotations

import asyncio
from unittest.mock import AsyncMock, MagicMock, call, patch

import pytest

from imas_codex.standard_names.attachment_audit import AttachmentAuditResult
from imas_codex.standard_names.budget import BudgetManager

_GO = "imas_codex.standard_names.graph_ops"
_AA = "imas_codex.standard_names.attachment_audit"
_DGO = "imas_codex.graph.dd_graph_ops"
_BM = "imas_codex.standard_names.budget.BudgetManager"
_LOOP = "imas_codex.standard_names.loop"


# ---------------------------------------------------------------------------
# --only → skip flags
# ---------------------------------------------------------------------------


def test_only_link_maps_to_skip_generate() -> None:
    from imas_codex.standard_names.turn import skip_flags_from_only

    flags = skip_flags_from_only("link")
    assert flags["skip_generate"] is True
    assert flags["skip_review"] is True


def test_only_review_maps_to_skip_generate() -> None:
    from imas_codex.standard_names.turn import skip_flags_from_only

    flags = skip_flags_from_only("review")
    assert flags["skip_generate"] is True
    # review phase keeps review pools running
    assert flags["skip_review"] is False


@pytest.mark.parametrize("selector", ["review_name", "refine_name"])
def test_exact_name_action_is_a_canonical_pool_selector(selector: str) -> None:
    from imas_codex.standard_names.turn import (
        TURN_PHASES,
        exact_pool_from_only,
        skip_flags_from_only,
    )

    assert selector in TURN_PHASES
    assert exact_pool_from_only(selector) == selector
    flags = skip_flags_from_only(selector)
    assert flags["skip_generate"] is True
    assert flags["skip_review"] is False


@pytest.mark.parametrize("selector", ["review", "review_names"])
def test_broad_review_selectors_remain_multi_pool(selector: str) -> None:
    from imas_codex.standard_names.turn import exact_pool_from_only

    assert exact_pool_from_only(selector) is None


def test_unknown_only_selector_fails() -> None:
    from imas_codex.standard_names.turn import exact_pool_from_only

    with pytest.raises(ValueError, match="unknown --only selector"):
        exact_pool_from_only("not_a_pool")


def test_attach_is_a_valid_phase_that_skips_generate() -> None:
    from imas_codex.standard_names.turn import TURN_PHASES, skip_flags_from_only

    assert "attach" in TURN_PHASES
    flags = skip_flags_from_only("attach")
    # attach runs no pools — the run_sn_pools short-circuit handles the focus.
    assert flags["skip_generate"] is True
    assert flags["skip_review"] is True


def test_only_reconcile_disables_every_pipeline_producer() -> None:
    from imas_codex.standard_names.turn import skip_flags_from_only

    flags = skip_flags_from_only("reconcile")
    assert flags == {
        "skip_generate": True,
        "skip_enrich": True,
        "skip_review": True,
        "skip_regen": True,
    }


@pytest.mark.parametrize(
    ("only_phase", "attach_only", "reconcile_only"),
    [
        ("reconcile", False, True),
        ("attach", True, False),
        ("link", False, False),
    ],
)
def test_cli_forwards_explicit_maintenance_mode(
    only_phase: str,
    attach_only: bool,
    reconcile_only: bool,
) -> None:
    """The selected maintenance command reaches the orchestrator unchanged."""
    from imas_codex.cli.sn import _run_sn_cmd

    run_pools = AsyncMock(return_value=MagicMock(stop_reason="completed"))

    def _run_discovery(_config, async_main):
        return asyncio.run(async_main(asyncio.Event(), MagicMock()))

    with (
        patch("imas_codex.cli.sn._require_embed_ready"),
        patch("imas_codex.cli.sn._require_terminal_drain"),
        patch(
            "imas_codex.cli.discover.common.use_rich_output",
            return_value=False,
        ),
        patch("imas_codex.cli.discover.common.setup_logging"),
        patch(
            "imas_codex.cli.discover.common.run_discovery",
            side_effect=_run_discovery,
        ),
        patch("imas_codex.standard_names.loop.run_sn_pools", new=run_pools),
        patch(
            "imas_codex.standard_names.loop.summary_table",
            return_value={"stop_reason": "completed"},
        ),
    ):
        _run_sn_cmd(
            cost_limit=5.0,
            per_domain_limit=None,
            dry_run=False,
            quiet=True,
            only=only_phase,
        )

    assert run_pools.await_args.kwargs["attach_only"] is attach_only
    assert run_pools.await_args.kwargs["reconcile_only"] is reconcile_only


# ---------------------------------------------------------------------------
# pool specs
# ---------------------------------------------------------------------------


def _build_specs(**kwargs) -> list:
    from imas_codex.standard_names.loop import _build_pool_specs

    mgr = BudgetManager(total_budget=10.0)
    stop = asyncio.Event()
    return _build_pool_specs(mgr, stop, **kwargs)


def test_build_pool_specs_skip_generate_drops_generate_name() -> None:
    """skip_generate removes the generate_name pool but keeps the rest."""
    names = {s.name for s in _build_specs(skip_generate=True)}
    assert "generate_name" not in names
    # review / refine / docs pools still run so link/review work drains
    assert "review_name" in names
    assert "refine_name" in names
    assert "generate_docs" in names


def test_build_pool_specs_default_keeps_generate_name() -> None:
    names = {s.name for s in _build_specs()}
    assert "generate_name" in names


@pytest.mark.parametrize("pool_name", ["review_name", "refine_name"])
def test_build_pool_specs_selects_exact_name_action(pool_name: str) -> None:
    names = {spec.name for spec in _build_specs(only_pool=pool_name)}

    assert names == {pool_name}


def test_build_pool_specs_rejects_unknown_pool() -> None:
    with pytest.raises(ValueError, match="unknown standard-name pool"):
        _build_specs(only_pool="review")


@pytest.mark.parametrize(
    ("pool_name", "filters"),
    [
        (
            "review_name",
            {"only_pool": "review_name", "names_only": True, "skip_generate": True},
        ),
        (
            "refine_name",
            {"only_pool": "refine_name", "names_only": True, "skip_generate": True},
        ),
        (
            "enrich_parents",
            {"names_only": True, "skip_generate": True, "skip_review": True},
        ),
    ],
)
def test_single_name_scope_keeps_one_replica(
    pool_name: str, filters: dict[str, object]
) -> None:
    recount = MagicMock(side_effect=AssertionError("exact scope recounted"))
    with (
        patch(f"{_LOOP}._count_scope_names", new=recount),
        patch("imas_codex.settings.get_pool_replicas", return_value=128),
    ):
        specs = _build_specs(
            scope_run_id="exact-scope",
            scope_size_hint=1,
            **filters,
        )

    assert [(spec.name, spec.replicas) for spec in specs] == [(pool_name, 1)]
    recount.assert_not_called()


def test_forty_name_scope_caps_every_pool_and_keeps_docs_stricter() -> None:
    recount = MagicMock(side_effect=AssertionError("exact scope recounted"))
    with (
        patch(f"{_LOOP}._count_scope_names", new=recount),
        patch("imas_codex.settings.get_pool_replicas", return_value=128),
    ):
        specs = _build_specs(
            scope_run_id="exact-scope",
            scope_size_hint=40,
        )

    replicas = {spec.name: spec.replicas for spec in specs}
    assert max(replicas.values()) <= 40
    assert replicas["generate_name"] == 40
    assert replicas["review_name"] == 40
    assert replicas["refine_name"] == 40
    assert replicas["enrich_parents"] == 40
    assert replicas["generate_docs"] == 20
    assert replicas["review_docs"] == 20
    assert replicas["refine_docs"] == 20
    recount.assert_not_called()


def test_scoped_fallback_counts_once_and_caps_safely() -> None:
    recount = MagicMock(return_value=3)
    with (
        patch(f"{_LOOP}._count_scope_names", new=recount),
        patch("imas_codex.settings.get_pool_replicas", return_value=128),
    ):
        specs = _build_specs(scope_run_id="focus-scope")

    replicas = {spec.name: spec.replicas for spec in specs}
    recount.assert_called_once_with("focus-scope", None)
    assert max(replicas.values()) <= 3
    assert replicas["review_name"] == 3
    assert replicas["enrich_parents"] == 3
    assert replicas["generate_docs"] == 2
    assert replicas["review_docs"] == 2
    assert replicas["refine_docs"] == 2


@pytest.mark.parametrize("invalid_hint", [0, -1, True, 1.5])
def test_invalid_scope_size_hint_fails_closed(invalid_hint: object) -> None:
    recount = MagicMock(side_effect=AssertionError("invalid scope recounted"))
    with (
        patch(f"{_LOOP}._count_scope_names", new=recount),
        pytest.raises(ValueError, match="positive integer"),
    ):
        _build_specs(
            scope_run_id="exact-scope",
            scope_size_hint=invalid_hint,
        )
    recount.assert_not_called()


def test_unscoped_scope_size_hint_fails_closed() -> None:
    recount = MagicMock(side_effect=AssertionError("unscoped hint recounted"))
    with (
        patch(f"{_LOOP}._count_scope_names", new=recount),
        pytest.raises(ValueError, match="requires a bounded graph scope"),
    ):
        _build_specs(scope_size_hint=1)
    recount.assert_not_called()


@pytest.mark.asyncio
async def test_review_then_refine_requires_separate_pool_selection() -> None:
    """One review quorum cannot cascade into refinement in the same action."""
    state = {
        "name_stage": "drafted",
        "edit_status": "open",
        "reviewer_score_name": None,
    }
    item = {"id": "edited_flux", "claim_token": "claim-token"}
    review_claim = MagicMock(return_value=[item])
    refine_claim = MagicMock()

    async def _review(items, _mgr, _stop_event, **_kwargs):
        assert items == [item]
        state["name_stage"] = "reviewed"
        state["reviewer_score_name"] = 0.2
        return 1

    async def _refine(items, _mgr, _stop_event, **_kwargs):
        assert items == [item]
        assert state["name_stage"] == "reviewed"
        assert state["edit_status"] == "open"
        state["name_stage"] = "drafted"
        return 1

    review_process = AsyncMock(side_effect=_review)
    refine_process = AsyncMock(side_effect=_refine)
    with (
        patch(
            f"{_LOOP}._count_scope_names",
            side_effect=AssertionError("exact scope recounted"),
        ),
        patch(f"{_GO}.claim_review_name_batch", new=review_claim),
        patch(f"{_GO}.claim_refine_name_batch", new=refine_claim),
        patch(
            "imas_codex.standard_names.workers.process_review_name_batch",
            new=review_process,
        ),
        patch(
            "imas_codex.standard_names.workers.process_refine_name_batch",
            new=refine_process,
        ),
    ):
        review_specs = _build_specs(
            only_pool="review_name",
            scope_run_id="exact-scope",
            scope_size_hint=1,
            edits_only=True,
            names_only=True,
            skip_generate=True,
        )
        assert [spec.name for spec in review_specs] == ["review_name"]
        review_batch = await review_specs[0].claim()
        assert review_batch is not None
        await review_specs[0].process(review_batch)

        assert state == {
            "name_stage": "reviewed",
            "edit_status": "open",
            "reviewer_score_name": 0.2,
        }
        refine_claim.assert_not_called()
        refine_process.assert_not_awaited()

        refine_claim.return_value = [item]
        refine_specs = _build_specs(
            only_pool="refine_name",
            scope_run_id="exact-scope",
            scope_size_hint=1,
            edits_only=True,
            names_only=True,
            skip_generate=True,
        )
        assert [spec.name for spec in refine_specs] == ["refine_name"]
        refine_batch = await refine_specs[0].claim()
        assert refine_batch is not None
        await refine_specs[0].process(refine_batch)

    review_claim.assert_called_once()
    review_process.assert_awaited_once()
    refine_claim.assert_called_once()
    refine_process.assert_awaited_once()
    assert state["name_stage"] == "drafted"


def test_run_command_forwards_exact_pool_selector() -> None:
    from imas_codex.cli.sn import _run_sn_cmd

    run_pools = AsyncMock(return_value=MagicMock(stop_reason="completed"))

    def _run_discovery(_config, async_main):
        return asyncio.run(async_main(asyncio.Event(), MagicMock()))

    with (
        patch("imas_codex.cli.sn._require_embed_ready"),
        patch("imas_codex.cli.sn._require_terminal_drain"),
        patch(
            "imas_codex.cli.discover.common.use_rich_output",
            return_value=False,
        ),
        patch("imas_codex.cli.discover.common.setup_logging"),
        patch(
            "imas_codex.cli.discover.common.run_discovery",
            side_effect=_run_discovery,
        ),
        patch("imas_codex.standard_names.loop.run_sn_pools", new=run_pools),
        patch(
            "imas_codex.standard_names.loop.summary_table",
            return_value={"stop_reason": "completed"},
        ),
    ):
        _run_sn_cmd(
            cost_limit=5.0,
            per_domain_limit=None,
            dry_run=False,
            quiet=True,
            only="review_name",
            scope_run_id="exact-scope",
            scope_size_hint=1,
        )

    assert run_pools.await_args.kwargs["only_pool"] == "review_name"
    assert run_pools.await_args.kwargs["scope_size_hint"] == 1


# ---------------------------------------------------------------------------
# auto-seed sweep
# ---------------------------------------------------------------------------


def _run_sn_pools_patches(seed_mock: AsyncMock):
    """Patch every graph-backed startup call so run_sn_pools reaches (or
    skips) the auto-seed branch without touching a live graph."""
    mock_gc_ctx = MagicMock()
    mock_gc_inst = MagicMock()
    mock_gc_inst.query.return_value = [{"cnt": 1}]
    mock_gc_ctx.__enter__ = MagicMock(return_value=mock_gc_inst)
    mock_gc_ctx.__exit__ = MagicMock(return_value=False)

    return [
        patch(f"{_GO}.reconcile_standard_name_sources", return_value={}),
        patch(f"{_GO}.reconcile_vocab_gaps", return_value={}),
        patch(
            f"{_GO}.revive_unit_skipped_sources",
            return_value={"checked": 0, "revived": 0},
        ),
        patch(
            f"{_GO}.retry_vocab_gap_sources_on_grammar_change",
            return_value={"checked": 0, "revived": 0},
        ),
        patch(f"{_GO}.reconcile_provenance", return_value={}),
        patch(f"{_GO}.reconcile_source_status_liveness", return_value={}),
        patch(f"{_GO}.retire_unreachable_hint_edits", return_value=0),
        patch(f"{_GO}.reconcile_grammar_segments", return_value={}),
        patch(f"{_GO}.reconcile_standard_name_cocos_links", return_value={}),
        patch(
            f"{_GO}.reconcile_standard_name_unit_edges",
            return_value={"names_realigned": 0, "edges_dropped": 0, "edges_created": 0},
        ),
        patch(
            f"{_AA}.reconcile_attachment_consistency",
            return_value=AttachmentAuditResult(),
        ),
        # The DD-unit correction reconcile opens its own GraphClient; stub it.
        patch(
            f"{_DGO}.reconcile_dd_unit_corrections",
            return_value={"checked": 0, "corrected": 0},
        ),
        patch(
            f"{_GO}.reconcile_standard_name_dd_edges",
            return_value={"edges_created": 0, "pairs_dropped": 0},
        ),
        patch(
            f"{_GO}.reconcile_standard_name_source_paths",
            return_value={"names_reconciled": 0},
        ),
        patch(
            f"{_GO}.reconcile_reviewable_name_stage",
            return_value={"names_advanced": 0},
        ),
        patch(f"{_GO}.create_sn_run_open"),
        patch(f"{_GO}.finalize_sn_run"),
        patch(f"{_GO}.release_all_orphan_claims", return_value={"sn": 0, "sns": 0}),
        patch(f"{_GO}.rederive_structural_edges", return_value={}),
        patch(f"{_GO}.seed_parent_sources", return_value=0),
        patch(f"{_GO}.normalize_derived_parent_lifecycle", return_value=0),
        patch(f"{_GO}.structural_accept_derived_parents", return_value=0),
        patch(f"{_GO}.reconcile_orphan_parent_sources", return_value=0),
        patch(f"{_GO}.resolve_doc_links", return_value={}),
        # The always-on stranded-reviewed promotion builds its own GraphClient;
        # mock it so the startup path stays graph-free.
        patch(f"{_GO}.promote_stranded_reviewed", return_value={"name": 0, "docs": 0}),
        # The always-on orphaned-SNRun sweep builds its own GraphClient; mock it.
        patch(f"{_GO}.mark_orphaned_standard_name_runs_stale", return_value=0),
        # The always-on source-drift refresh builds its own GraphClient at the
        # source_refresh binding site, which the graph.client patch below does
        # not intercept once that module is already imported. Mock the refresh
        # itself so the startup path stays graph-free regardless of import order.
        patch(
            "imas_codex.standard_names.source_refresh.refresh_drifted_sources",
            return_value={},
        ),
        patch(f"{_LOOP}._seed_all_domains", new=seed_mock),
        patch(
            "imas_codex.standard_names.pools.run_pools",
            new_callable=AsyncMock,
            return_value={},
        ),
        patch(f"{_BM}.start", new_callable=AsyncMock),
        patch(f"{_BM}.drain_pending", new_callable=AsyncMock, return_value=True),
        patch(f"{_BM}.get_total_spent", new_callable=AsyncMock, return_value=0.0),
        patch(f"{_BM}.exhausted", return_value=True),
        patch(f"{_BM}.phase_spent", new_callable=lambda: property(lambda self: {})),
        patch("imas_codex.graph.client.GraphClient", return_value=mock_gc_ctx),
    ]


async def _run(skip_generate: bool) -> AsyncMock:
    from imas_codex.standard_names.loop import run_sn_pools

    seed_mock = AsyncMock(return_value=0)
    patches = _run_sn_pools_patches(seed_mock)
    for p in patches:
        p.start()
    try:
        stop = asyncio.Event()
        stop.set()  # immediate stop — we only care about the startup path
        await run_sn_pools(
            cost_limit=5.0,
            domains=(),  # empty → auto-seed unless suppressed
            stop_event=stop,
            skip_generate=skip_generate,
        )
    finally:
        for p in patches:
            p.stop()
    return seed_mock


@pytest.mark.asyncio
async def test_run_sn_pools_skip_generate_skips_autoseed() -> None:
    """With skip_generate, the domain auto-seed sweep must not run."""
    seed_mock = await _run(skip_generate=True)
    assert seed_mock.await_count == 0


@pytest.mark.asyncio
async def test_run_sn_pools_without_skip_generate_autoseeds() -> None:
    """Control: without skip_generate, the auto-seed sweep runs (domains=())."""
    seed_mock = await _run(skip_generate=False)
    assert seed_mock.await_count == 1


@pytest.mark.asyncio
async def test_reconcile_only_finishes_parent_maintenance_without_pools() -> None:
    """Reconciliation includes parent repair but has no operational workers."""
    from imas_codex.standard_names.loop import run_sn_pools

    seed_mock = AsyncMock(return_value=0)
    patches = _run_sn_pools_patches(seed_mock)
    rederive_structural_edges = MagicMock(return_value={})
    seed_parent_sources = MagicMock(return_value=0)
    normalize_parent_lifecycle = MagicMock(return_value=5)
    structural_accept_parents = MagicMock(return_value=0)
    reconcile_orphan_parents = MagicMock(return_value=0)
    build_specs = MagicMock(side_effect=AssertionError("pool specs constructed"))
    run_pools = AsyncMock(side_effect=AssertionError("worker pools started"))
    run_orphan_sweep = AsyncMock(side_effect=AssertionError("worker started"))
    embed_descriptions = AsyncMock(side_effect=AssertionError("worker started"))
    call_llm = MagicMock(side_effect=AssertionError("LLM called"))
    acall_llm = AsyncMock(side_effect=AssertionError("LLM called"))
    patches.extend(
        [
            patch(
                f"{_GO}.rederive_structural_edges",
                new=rederive_structural_edges,
            ),
            patch(f"{_GO}.seed_parent_sources", new=seed_parent_sources),
            patch(
                f"{_GO}.normalize_derived_parent_lifecycle",
                new=normalize_parent_lifecycle,
            ),
            patch(
                f"{_GO}.structural_accept_derived_parents",
                new=structural_accept_parents,
            ),
            patch(
                f"{_GO}.reconcile_orphan_parent_sources",
                new=reconcile_orphan_parents,
            ),
            patch(f"{_LOOP}._build_pool_specs", new=build_specs),
            patch("imas_codex.standard_names.pools.run_pools", new=run_pools),
            patch(
                "imas_codex.standard_names.orphan_sweep.run_orphan_sweep_loop",
                new=run_orphan_sweep,
            ),
            patch(
                "imas_codex.discovery.base.embed_worker.embed_description_worker",
                new=embed_descriptions,
            ),
            patch(
                "imas_codex.discovery.base.llm.call_llm_structured",
                new=call_llm,
            ),
            patch(
                "imas_codex.discovery.base.llm.acall_llm_structured",
                new=acall_llm,
            ),
        ]
    )
    for item in patches:
        item.start()
    try:
        summary = await run_sn_pools(
            cost_limit=5.0,
            domains=(),
            skip_generate=True,
            skip_review=True,
            reconcile_only=True,
        )
    finally:
        for item in reversed(patches):
            item.stop()

    rederive_structural_edges.assert_called_once()
    seed_parent_sources.assert_called_once()
    normalize_parent_lifecycle.assert_called_once()
    structural_accept_parents.assert_called_once()
    reconcile_orphan_parents.assert_called_once()
    seed_mock.assert_not_awaited()
    build_specs.assert_not_called()
    run_pools.assert_not_awaited()
    run_orphan_sweep.assert_not_awaited()
    embed_descriptions.assert_not_awaited()
    call_llm.assert_not_called()
    acall_llm.assert_not_awaited()
    assert summary.cost_spent == 0.0
    assert summary.stop_reason == "completed"


@pytest.mark.asyncio
async def test_operational_run_repeats_structural_maintenance_after_drain() -> None:
    """Operational runs repair structures both before and after pool work."""
    from imas_codex.standard_names.loop import run_sn_pools

    seed_mock = AsyncMock(return_value=0)
    patches = _run_sn_pools_patches(seed_mock)
    maintenance = MagicMock()
    maintenance.rederive.return_value = {}
    maintenance.seed.return_value = 0
    maintenance.normalize.return_value = 0
    maintenance.reconcile_orphans.return_value = 0
    patches.extend(
        [
            patch(
                f"{_GO}.rederive_structural_edges",
                new=maintenance.rederive,
            ),
            patch(f"{_GO}.seed_parent_sources", new=maintenance.seed),
            patch(
                f"{_GO}.normalize_derived_parent_lifecycle",
                new=maintenance.normalize,
            ),
            patch(
                f"{_GO}.reconcile_orphan_parent_sources",
                new=maintenance.reconcile_orphans,
            ),
        ]
    )
    for item in patches:
        item.start()
    try:
        stop = asyncio.Event()
        stop.set()
        await run_sn_pools(
            cost_limit=5.0,
            domains=(),
            stop_event=stop,
            skip_generate=True,
        )
    finally:
        for item in reversed(patches):
            item.stop()

    assert maintenance.mock_calls == [
        call.rederive(),
        call.seed(),
        call.normalize(),
        call.reconcile_orphans(),
        call.rederive(),
        call.normalize(),
        call.reconcile_orphans(),
    ]


@pytest.mark.asyncio
async def test_attach_only_runs_two_reconciles_and_no_pools() -> None:
    """``attach_only`` short-circuits to the DD-edge + source_paths reconcile.

    It must run exactly those two reconciles, never seed domains, and never
    launch the pools — the focused, no-LLM one-shot backfill.
    """
    from imas_codex.standard_names.loop import run_sn_pools

    seed_mock = AsyncMock(return_value=0)
    with (
        patch(
            f"{_GO}.reconcile_standard_name_dd_edges",
            return_value={"edges_created": 7, "pairs_dropped": 2},
        ) as dd_edge,
        patch(
            f"{_GO}.reconcile_standard_name_source_paths",
            return_value={"names_reconciled": 3},
        ) as sp,
        patch(
            f"{_GO}.reconcile_reviewable_name_stage",
            return_value={"names_advanced": 0},
        ),
        patch(f"{_LOOP}._seed_all_domains", new=seed_mock),
        patch(
            "imas_codex.standard_names.pools.run_pools",
            new_callable=AsyncMock,
            return_value={},
        ) as run_pools_mock,
    ):
        summary = await run_sn_pools(cost_limit=5.0, attach_only=True)

    dd_edge.assert_called_once()
    sp.assert_called_once()
    seed_mock.assert_not_awaited()
    run_pools_mock.assert_not_awaited()
    assert summary.sources_reconciled == 7
    assert summary.stop_reason == "completed"
