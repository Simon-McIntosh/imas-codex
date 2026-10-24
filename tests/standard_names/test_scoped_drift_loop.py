"""A scoped pool run confines its source-drift refresh to its own names.

The refresh itself is guarded elsewhere. What no test reached before is the
loop half: :func:`_scope_name_ids` decides which names a scoped run owns, and
``run_sn_pools`` forwards that set to :func:`refresh_drifted_sources`. Drop
either half and a family apply sweeps the whole graph again, steering names no
one asked it to touch.

``_scope_name_ids`` also has to fail closed in two different ways. A graph it
cannot reach must skip the refresh and say so, never read as "this run owns
nothing" and never fall open into an unscoped sweep. A ``RuntimeError`` it was
never meant to catch — the default-tier guard in ``conftest.py`` raises one
when a test opens a live ``GraphClient`` — must propagate, or the guard reports
a pass it never inspected.
"""

from __future__ import annotations

import asyncio
import logging
from contextlib import ExitStack
from typing import Any
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from imas_codex.standard_names.attachment_audit import AttachmentAuditResult

_GO = "imas_codex.standard_names.graph_ops"
_LOOP = "imas_codex.standard_names.loop"
_REFRESH = "imas_codex.standard_names.source_refresh.refresh_drifted_sources"

_SCOPE_IDS_CYPHER = "RETURN DISTINCT sn.id AS id"


def _graph_context(scope_ids: list[str] | None) -> MagicMock:
    """A ``GraphClient`` double whose scope query answers *scope_ids*.

    Every other graph read in ``run_sn_pools`` is patched out, so the only
    reader that reaches this double is the scope selector.
    """
    graph = MagicMock()

    def _query(cypher: str, **kwargs: Any) -> list[dict[str, Any]]:
        if _SCOPE_IDS_CYPHER in cypher:
            return [{"id": nid} for nid in (scope_ids or [])]
        return [{"cnt": 1}]

    graph.query.side_effect = _query
    context = MagicMock()
    context.__enter__.return_value = graph
    context.__exit__.return_value = False
    return context


def _maintenance_mocks(stack: ExitStack) -> MagicMock:
    """Patch the graph-wide maintenance call set used by ``run_sn_pools``."""
    specs: dict[str, object] = {
        "mark_orphaned_standard_name_runs_stale": 0,
        "reconcile_standard_name_sources": {},
        "reconcile_vocab_gaps": {},
        "revive_unit_skipped_sources": {},
        "retry_vocab_gap_sources_on_grammar_change": {},
        "reconcile_provenance": {},
        "reconcile_source_status_liveness": {},
        "retire_unreachable_hint_edits": 0,
        "reconcile_grammar_segments": {},
        "reconcile_catalog_status": {},
        "reconcile_reviewable_name_stage": {},
        "reconcile_standard_name_cocos_links": {},
        "reconcile_standard_name_unit_edges": {},
        "reconcile_standard_name_dd_edges": {},
        "reconcile_standard_name_source_paths": {},
        "promote_stranded_reviewed": {},
        "rederive_structural_edges": {},
        "seed_parent_sources": 0,
        "normalize_derived_parent_lifecycle": 0,
        "structural_accept_derived_parents": 0,
        "reconcile_orphan_parent_sources": 0,
        "release_all_orphan_claims": {},
        "resolve_doc_links": {},
    }
    for name, result in specs.items():
        stack.enter_context(patch(f"{_GO}.{name}", return_value=result))
    stack.enter_context(
        patch(
            "imas_codex.graph.dd_graph_ops.reconcile_dd_unit_corrections",
            return_value={},
        )
    )
    refresh = stack.enter_context(patch(_REFRESH, return_value={}))
    stack.enter_context(
        patch(
            "imas_codex.standard_names.harmonize.restamp_harmonized_families",
            return_value={},
        )
    )
    stack.enter_context(
        patch(
            "imas_codex.standard_names.orphan_sweep.run_orphan_sweep_loop",
            new_callable=AsyncMock,
        )
    )
    stack.enter_context(
        patch(
            "imas_codex.discovery.base.embed_worker.embed_description_worker",
            new_callable=AsyncMock,
        )
    )
    return refresh


def _run_stack(stack: ExitStack, graph_context: MagicMock) -> None:
    """Patch the remaining collaborators of one bounded ``run_sn_pools`` run."""
    stack.enter_context(
        patch(
            "imas_codex.standard_names.attachment_audit."
            "reconcile_attachment_consistency",
            return_value=AttachmentAuditResult(),
        )
    )
    stack.enter_context(patch(f"{_GO}.create_sn_run_open"))
    stack.enter_context(patch(f"{_GO}.finalize_sn_run"))
    stack.enter_context(
        patch(
            f"{_GO}.scoped_terminal_residue",
            return_value={"total": 0, "names": [], "sources": []},
        )
    )
    stack.enter_context(
        patch(
            "imas_codex.standard_names.ledger.find_provenance_orphans",
            return_value=[],
        )
    )
    stack.enter_context(
        patch(
            "imas_codex.standard_names.audits.find_flux_surface_reduction_violations",
            return_value=[],
        )
    )
    stack.enter_context(
        patch(
            "imas_codex.standard_names.audits.find_removed_dd_sources",
            return_value=[],
        )
    )
    stack.enter_context(patch(f"{_GO}.persist_outcome_snapshot", return_value={}))
    stack.enter_context(patch(f"{_GO}.reset_persist_outcomes"))
    stack.enter_context(patch(f"{_LOOP}._build_pool_specs", return_value=[]))
    stack.enter_context(
        patch(
            "imas_codex.standard_names.pools.run_pools",
            new_callable=AsyncMock,
            return_value={},
        )
    )
    stack.enter_context(
        patch(
            "imas_codex.standard_names.budget.BudgetManager.start",
            new_callable=AsyncMock,
        )
    )
    stack.enter_context(
        patch(
            "imas_codex.standard_names.budget.BudgetManager.drain_pending",
            new_callable=AsyncMock,
            return_value=True,
        )
    )
    stack.enter_context(
        patch(
            "imas_codex.standard_names.budget.BudgetManager._get_total_spent_sync",
            return_value=0.0,
        )
    )
    stack.enter_context(
        patch("imas_codex.graph.client.GraphClient", return_value=graph_context)
    )


def _run_loop(
    scope_run_id: str | None, scope_ids: list[str] | None
) -> tuple[Any, MagicMock]:
    from imas_codex.standard_names.loop import run_sn_pools

    with ExitStack() as stack:
        refresh = _maintenance_mocks(stack)
        _run_stack(stack, _graph_context(scope_ids))
        stop = asyncio.Event()
        stop.set()
        summary = asyncio.run(
            run_sn_pools(
                cost_limit=5.0,
                scope_run_id=scope_run_id,
                stop_event=stop,
                skip_global_maintenance=False,
            )
        )
    return summary, refresh


def test_scoped_loop_forwards_its_own_name_set_to_the_refresh() -> None:
    """Dropping ``sn_ids=_refresh_ids`` from the call lets the refresh go wide."""
    summary, refresh = _run_loop("scope-1", ["alpha", "beta"])

    refresh.assert_called_once()
    kwargs = refresh.call_args.kwargs
    assert kwargs["sn_ids"] == ["alpha", "beta"]
    assert kwargs["scope_run_id"] == summary.run_id


def test_unscoped_loop_leaves_the_refresh_graph_wide() -> None:
    _, refresh = _run_loop(None, None)

    refresh.assert_called_once()
    assert refresh.call_args.kwargs["sn_ids"] is None


def test_a_scope_that_owns_nothing_refreshes_nothing() -> None:
    """An empty readable scope is a target set of no names, not a wide sweep."""
    _, refresh = _run_loop("scope-1", [])

    refresh.assert_called_once()
    assert refresh.call_args.kwargs["sn_ids"] == []


def test_an_unreadable_scope_skips_the_refresh(
    caplog: pytest.LogCaptureFixture,
) -> None:
    """A failed scope read skips the refresh: it must neither sweep nor read as empty."""
    from neo4j.exceptions import ServiceUnavailable

    from imas_codex.standard_names import loop

    unreadable = ServiceUnavailable("no route to the graph")
    with (
        caplog.at_level(logging.WARNING),
        patch("imas_codex.graph.client.GraphClient", side_effect=unreadable),
    ):
        assert loop._scope_name_ids("scope-1") is None
    assert "scope-1" in caplog.text

    with patch.object(loop, "_scope_name_ids", return_value=None):
        _, refresh = _run_loop("scope-1", ["alpha"])

    refresh.assert_not_called()


def test_scope_selector_reads_both_ways_a_run_binds_its_names() -> None:
    """The selector must ask for direct-marked and source-produced names.

    Dropping either arm silently shrinks a scope: a family apply seeds its
    members through ``StandardNameSource`` and marks only some directly, so
    one arm alone returns the wrong set.
    """
    from imas_codex.standard_names import loop

    captured: dict[str, Any] = {}

    class _Graph:
        def query(self, cypher: str, **kwargs: Any) -> list[dict[str, str]]:
            captured["cypher"] = cypher
            captured["kwargs"] = kwargs
            return [{"id": "from-run"}, {"id": "from-source"}]

    class _Client:
        def __enter__(self) -> _Graph:
            return _Graph()

        def __exit__(self, *exc: object) -> bool:
            return False

    with patch("imas_codex.graph.client.GraphClient", _Client):
        ids = loop._scope_name_ids("scope-1")

    assert ids == ["from-run", "from-source"]
    assert captured["kwargs"] == {"rid": "scope-1"}
    assert "sn.run_id = $rid" in captured["cypher"]
    assert "StandardNameSource {run_id: $rid}" in captured["cypher"]
    assert "PRODUCED_NAME" in captured["cypher"]


def test_scope_selector_lets_the_default_tier_live_graph_guard_raise() -> None:
    """The selector's error handling must not swallow the conftest guard.

    ``conftest.py`` raises ``RuntimeError`` from ``GraphClient.__post_init__``
    when a default-tier test opens a live client. Catch-everything here turns
    that into an empty scope, so the guard's refusal reads as a passing test.
    """
    from imas_codex.standard_names import loop

    with pytest.raises(RuntimeError, match="default-tier test"):
        loop._scope_name_ids("scope-1")
