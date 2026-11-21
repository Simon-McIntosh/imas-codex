"""A run given a scope changes nothing outside it.

The pipeline's graph-wide maintenance passes rewrite rows the caller never
named.  A review of one edit's names used to run them over the whole
catalogue; the scope is now resolved once, at the CLI boundary, and the
orchestrator receives the bypass whether or not the operator repeated the
explicit flag.

The default-tier tests pin the decision at that boundary and at the
orchestrator entrance.  The graph-marked test plants an out-of-scope name in a
state the source reconciliation acts on and digests its properties and
relationships before and after a scoped run; a control arm runs the same
maintenance directly and shows the digest does change, so an unchanged digest
after the run is evidence that the pass never reached the name.
"""

from __future__ import annotations

import asyncio
import hashlib
import json
import uuid
from contextlib import ExitStack
from typing import Any
from unittest.mock import AsyncMock, MagicMock, patch

import pytest
from click.testing import CliRunner

from iter_codex.cli.sn import _run_sn_cmd, sn
from iter_codex.standard_names.attachment_audit import AttachmentAuditResult
from iter_codex.standard_names.graph_store import (
    owner_module_path as _owner_module_path,
)
from iter_codex.standard_names.pipeline import resolve_scope_maintenance

_FOCUS_PATH = "equilibrium/time_slice/profiles_1d/psi"
_LOOP = "iter_codex.standard_names.loop"

# ---------------------------------------------------------------------------
# The policy, judged on its own
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    "kwargs",
    [
        {"scope_run_id": "sn-edit-probe"},
        {"focus_paths": (_FOCUS_PATH,)},
        {"exact_names": ("poloidal_flux",)},
        {"edits_only": True},
    ],
    ids=["run-id", "focus", "exact-name", "edits"],
)
def test_any_scope_bypasses_global_maintenance(
    kwargs: dict[str, Any],
) -> None:
    policy = resolve_scope_maintenance(**kwargs)

    assert policy.scoped is True
    assert policy.bypass_global_maintenance is True
    assert policy.refusal is None


def test_an_unscoped_run_keeps_global_maintenance() -> None:
    policy = resolve_scope_maintenance()

    assert policy.scoped is False
    assert policy.bypass_global_maintenance is False
    assert policy.refusal is None


@pytest.mark.parametrize(
    "kwargs, fragment",
    [
        ({"scope_run_id": "r", "only_phase": "reconcile"}, "graph-wide"),
        ({"scope_run_id": "r", "only_phase": "attach"}, "graph-wide"),
        ({"scope_run_id": "r", "reset_to": "drafted"}, "--reset-to"),
        ({"scope_run_id": "r", "revalidate": True}, "--revalidate"),
    ],
)
def test_a_scope_refuses_actions_it_cannot_bound(
    kwargs: dict[str, Any], fragment: str
) -> None:
    policy = resolve_scope_maintenance(**kwargs)

    assert policy.scoped is True
    assert policy.bypass_global_maintenance is False
    assert policy.refusal is not None
    assert fragment in policy.refusal


def test_reset_to_with_focus_paths_is_bounded_and_still_bypasses() -> None:
    policy = resolve_scope_maintenance(focus_paths=(_FOCUS_PATH,), reset_to="drafted")

    assert policy.bypass_global_maintenance is True
    assert policy.refusal is None


# ---------------------------------------------------------------------------
# The CLI boundary and the orchestrator entrance
# ---------------------------------------------------------------------------


def _drive_run_discovery(_config, async_main):
    return asyncio.run(async_main(asyncio.Event(), MagicMock()))


def _invoke_sn_run(args: list[str]):
    """Drive ``sn run`` with the discovery harness and service probes stubbed."""
    pools = AsyncMock(return_value=MagicMock(stop_reason="completed"))
    with (
        patch("iter_codex.cli.sn._require_embed_ready"),
        patch("iter_codex.cli.sn._require_terminal_drain"),
        patch(
            "iter_codex.cli.sn._configured_local_compose_required",
            return_value=False,
        ),
        patch("iter_codex.cli.sn._auto_sync_grammar"),
        patch("iter_codex.cli.sn._note_pipeline_version_drift"),
        patch("iter_codex.cli.discover.common.use_rich_output", return_value=False),
        patch("iter_codex.cli.discover.common.setup_logging"),
        patch(
            "iter_codex.cli.discover.common.run_discovery",
            side_effect=_drive_run_discovery,
        ),
        patch("iter_codex.standard_names.loop.run_sn_pools", new=pools),
        patch(
            "iter_codex.standard_names.loop.summary_table",
            return_value={"stop_reason": "completed"},
        ),
    ):
        result = CliRunner().invoke(sn, ["run", *args])
    return result, pools


def test_edits_scoped_review_forwards_the_bypass_without_the_flag() -> None:
    result, pools = _invoke_sn_run(
        [
            "--scope-run-id",
            "sn-edit-probe",
            "--edits",
            "--only",
            "review_name",
            "--quiet",
        ]
    )

    assert result.exit_code == 0, result.output
    pools.assert_awaited_once()
    kwargs = pools.await_args.kwargs
    # Known-present controls: the recorder reached the orchestrator seam with
    # the edit's own scope before the maintenance decision is judged.
    assert kwargs["scope_run_id"] == "sn-edit-probe"
    assert kwargs["edits_only"] is True
    assert kwargs["skip_global_maintenance"] is True


def test_unscoped_run_forwards_no_bypass() -> None:
    result, pools = _invoke_sn_run(["--quiet"])

    assert result.exit_code == 0, result.output
    pools.assert_awaited_once()
    assert pools.await_args.kwargs["skip_global_maintenance"] is False


def _call_run_sn_cmd(**overrides: Any):
    """Call the orchestrator entrance directly, with its harness stubbed."""
    pools = AsyncMock(return_value=MagicMock(stop_reason="completed"))
    kwargs: dict[str, Any] = {
        "cost_limit": 1.0,
        "per_domain_limit": None,
        "dry_run": False,
        "quiet": True,
        "skip_global_maintenance": False,
    }
    kwargs.update(overrides)
    with (
        patch(
            "iter_codex.cli.sn._configured_local_compose_required",
            return_value=False,
        ),
        patch("iter_codex.cli.sn._require_terminal_drain"),
        patch("iter_codex.cli.discover.common.use_rich_output", return_value=False),
        patch("iter_codex.cli.discover.common.setup_logging"),
        patch(
            "iter_codex.cli.discover.common.run_discovery",
            side_effect=_drive_run_discovery,
        ),
        patch("iter_codex.standard_names.loop.run_sn_pools", new=pools),
        patch(
            "iter_codex.standard_names.loop.summary_table",
            return_value={"stop_reason": "completed"},
        ),
    ):
        row = _run_sn_cmd(**kwargs)
    return row, pools


def test_orchestrator_entrance_forces_the_bypass_for_a_direct_caller() -> None:
    """A caller that hands over a scope and no flag still cannot widen the run."""
    _row, pools = _call_run_sn_cmd(scope_run_id="sn-edit-probe")

    pools.assert_awaited_once()
    kwargs = pools.await_args.kwargs
    assert kwargs["scope_run_id"] == "sn-edit-probe"
    assert kwargs["skip_global_maintenance"] is True


def test_orchestrator_entrance_leaves_an_unscoped_run_whole() -> None:
    _row, pools = _call_run_sn_cmd()

    pools.assert_awaited_once()
    assert pools.await_args.kwargs["skip_global_maintenance"] is False


def test_orchestrator_entrance_refuses_a_graph_wide_phase_under_a_scope() -> None:
    """The entrance enforces the same refusal the command does."""
    with pytest.raises(ValueError, match="graph-wide maintenance pass"):
        _call_run_sn_cmd(scope_run_id="sn-edit-probe", only="reconcile")


class _ReachedTheLoopGuard(Exception):
    """Raised from the first statement the loop runs after its guard."""


def _raise_after_guard(**_kwargs: Any) -> Any:
    raise _ReachedTheLoopGuard


def _edit_graph(ids: list[str]) -> tuple[MagicMock, MagicMock]:
    """A graph fake whose edit-scope read returns *ids*."""
    graph = MagicMock()

    def _query(statement: Any, **_: Any) -> list[dict[str, Any]]:
        if "edit_status" in str(statement):
            return [{"id": name_id} for name_id in ids]
        return [{"cnt": 1}]

    graph.query.side_effect = _query
    context = MagicMock()
    context.__enter__.return_value = graph
    context.__exit__.return_value = False
    return context, graph


def _call_run_sn_cmd_to_the_loop(
    *,
    open_edit_ids: list[str],
) -> tuple[Any, list[dict[str, Any]], str | None, BaseException | None]:
    """Drive the real entrance into the real loop, stopped just past its guard.

    ``run_sn_pools`` is *not* stubbed: its first act is the scope guard that
    rejected an edits-only run before this fix.  The loop is stopped by
    replacing the summary it builds immediately after the guard, so reaching
    the sentinel is proof the guard accepted the run.
    """
    graph_context, _graph = _edit_graph(open_edit_ids)
    bindings: list[dict[str, Any]] = []

    def _bind(name_ids: Any, run_id: str, **_: Any) -> dict[str, Any]:
        bindings.append({"ids": list(name_ids), "run_id": run_id})
        return {"name_ids": list(name_ids), "run_id": run_id, "stamped": len(name_ids)}

    row: Any = None
    error: BaseException | None = None
    with (
        patch(
            "iter_codex.cli.sn._configured_local_compose_required",
            return_value=False,
        ),
        patch("iter_codex.cli.sn._require_terminal_drain"),
        patch("iter_codex.cli.discover.common.use_rich_output", return_value=False),
        patch("iter_codex.cli.discover.common.setup_logging"),
        patch(
            "iter_codex.cli.discover.common.run_discovery",
            side_effect=_drive_run_discovery,
        ),
        patch(
            "iter_codex.standard_names.loop.summary_table",
            return_value={"stop_reason": "completed"},
        ),
        patch("iter_codex.graph.client.GraphClient", return_value=graph_context),
        patch(
            "iter_codex.standard_names.graph_store.graph_claim_scope."
            "scope_exact_standard_names",
            side_effect=_bind,
        ),
        patch(
            "iter_codex.standard_names.loop.RunSummary",
            new=_raise_after_guard,
        ),
    ):
        try:
            row = _run_sn_cmd(
                cost_limit=1.0,
                per_domain_limit=None,
                dry_run=False,
                quiet=True,
                skip_global_maintenance=False,
                edits_only=True,
            )
        except _ReachedTheLoopGuard as exc:  # the guard let the run through
            error = exc
    return row, bindings, None, error


def test_edits_only_run_binds_its_scope_and_passes_the_loop_guard() -> None:
    """``sn run --edits`` alone starts scoped instead of aborting at the guard."""
    _row, bindings, _scope, error = _call_run_sn_cmd_to_the_loop(
        open_edit_ids=["name_a", "name_b"]
    )

    assert isinstance(error, _ReachedTheLoopGuard), (
        "the loop's scope guard rejected the edits-only run"
    )
    assert bindings, "the entrance never bound the pending edit successors"
    assert bindings[0]["ids"] == ["name_a", "name_b"]
    # The same run id the scope was bound to is the one the loop must receive.
    assert bindings[0]["run_id"]


def test_edits_only_run_with_no_pending_successors_is_a_no_op() -> None:
    row, bindings, _scope, error = _call_run_sn_cmd_to_the_loop(open_edit_ids=[])

    assert row is None
    assert error is None
    assert bindings == []


def _graph_context() -> tuple[MagicMock, MagicMock]:
    graph = MagicMock()
    graph.query.return_value = [{"cnt": 1}]
    context = MagicMock()
    context.__enter__.return_value = graph
    context.__exit__.return_value = False
    return context, graph


def test_focused_scoped_run_leaves_other_runs_stamps_alone() -> None:
    graph_context, graph = _graph_context()
    with (
        patch("iter_codex.cli.sn._require_embed_ready"),
        patch("iter_codex.graph.client.GraphClient", return_value=graph_context),
        patch(
            "iter_codex.standard_names.graph_store.graph_claim_scope."
            "partition_focus_by_accepted",
            return_value=([_FOCUS_PATH], []),
        ),
        patch(
            "iter_codex.standard_names.graph_store.graph_source_bindings."
            "merge_standard_name_sources",
            return_value=1,
        ),
        patch(
            "iter_codex.standard_names.graph_store.graph_claim_scope.scope_focus_names"
        ),
        patch("iter_codex.settings.get_dd_version", return_value="4.0.0"),
        patch("iter_codex.cli.sn._run_sn_cmd", return_value={}) as run_cmd,
    ):
        result = CliRunner().invoke(sn, ["run", "--focus", _FOCUS_PATH, "--quiet"])

    assert result.exit_code == 0, result.output
    run_cmd.assert_called_once()
    assert run_cmd.call_args.kwargs["scope_run_id"] is not None
    queries = [str(call.args[0]) for call in graph.query.call_args_list if call.args]
    # The stale-stamp clear stays inside the named paths: no whole-catalogue
    # statement, and the statement that does run carries this run's sources.
    assert not any(
        "MATCH (sn:StandardName) WHERE sn.run_id IS NOT NULL" in query
        for query in queries
    )
    scoped_clears = [
        call
        for call in graph.query.call_args_list
        if call.args
        and "FOREACH (sn IN names | SET sn.run_id = NULL)" in str(call.args[0])
    ]
    assert scoped_clears, "no scope-bounded run_id clear was issued"
    assert scoped_clears[0].kwargs["source_ids"] == [f"dd:{_FOCUS_PATH}"]


@pytest.mark.parametrize(
    "args, fragment",
    [
        (
            ["--scope-run-id", "r", "--only", "reconcile"],
            "graph-wide maintenance",
        ),
        (["--scope-run-id", "r", "--reset-to", "drafted"], "--reset-to"),
        (["--focus", _FOCUS_PATH, "--revalidate"], "--revalidate"),
    ],
)
def test_cli_refuses_a_scope_it_cannot_bound(args: list[str], fragment: str) -> None:
    result = CliRunner().invoke(sn, ["run", *args])

    assert result.exit_code == 2
    assert fragment in result.output


# ---------------------------------------------------------------------------
# A recorded scoped run on a disposable graph
# ---------------------------------------------------------------------------

#: The loop's complete graph-wide maintenance set, silenced for this test so
#: exactly one writer can act on the disposable graph: the source
#: reconciliation the planted name is staged for.  Copied rather than imported
#: so a change to the loop's set cannot silently widen this test's arm.
_OTHER_MAINTENANCE_SET = {
    "mark_orphaned_standard_name_runs_stale": 0,
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


def _silence_other_maintenance(stack: ExitStack) -> dict[str, MagicMock]:
    mocks = {
        name: stack.enter_context(
            patch(f"{_owner_module_path(name)}.{name}", return_value=result)
        )
        for name, result in _OTHER_MAINTENANCE_SET.items()
    }
    mocks["reconcile_attachment_consistency"] = stack.enter_context(
        patch(
            "iter_codex.standard_names.attachment_audit."
            "reconcile_attachment_consistency",
            return_value=AttachmentAuditResult(),
        )
    )
    mocks["run_orphan_sweep_loop"] = stack.enter_context(
        patch(
            "iter_codex.standard_names.orphan_sweep.run_orphan_sweep_loop",
            new_callable=AsyncMock,
        )
    )
    mocks["embed_description_worker"] = stack.enter_context(
        patch(
            "iter_codex.discovery.base.embed_worker.embed_description_worker",
            new_callable=AsyncMock,
        )
    )
    return mocks


async def _run_scoped_pools(scope_run_id: str) -> Any:
    """Run the real orchestrator for one scope, with its workers stubbed.

    Every graph-wide writer except ``reconcile_standard_name_sources`` is
    silenced, so the run's reach over the planted names is exactly the
    maintenance decision under test.
    """
    from iter_codex.standard_names.budget import BudgetManager
    from iter_codex.standard_names.loop import run_sn_pools

    policy = resolve_scope_maintenance(scope_run_id=scope_run_id)
    with ExitStack() as stack:
        _silence_other_maintenance(stack)
        stack.enter_context(
            patch(
                "iter_codex.standard_names.pools.run_pools",
                new_callable=AsyncMock,
                return_value={},
            )
        )
        stack.enter_context(patch(f"{_LOOP}._build_pool_specs", return_value=[]))
        stack.enter_context(
            patch(
                "iter_codex.standard_names.graph_store.graph_parent_lifecycle."
                "scoped_terminal_residue",
                return_value={"total": 0, "names": [], "sources": []},
            )
        )
        stack.enter_context(
            patch(
                "iter_codex.standard_names.ledger.find_provenance_orphans",
                return_value=[],
            )
        )
        stack.enter_context(
            patch(
                "iter_codex.standard_names.audits."
                "find_flux_surface_reduction_violations",
                return_value=[],
            )
        )
        stack.enter_context(
            patch(
                "iter_codex.standard_names.audits.find_removed_dd_sources",
                return_value=[],
            )
        )
        stack.enter_context(
            patch(
                "iter_codex.standard_names.graph_store.graph_source_bindings."
                "persist_outcome_snapshot",
                return_value={},
            )
        )
        stack.enter_context(
            patch(
                "iter_codex.standard_names.graph_store.graph_source_bindings."
                "reset_persist_outcomes"
            )
        )
        stack.enter_context(
            patch.object(BudgetManager, "start", new_callable=AsyncMock)
        )
        stack.enter_context(
            patch.object(
                BudgetManager,
                "drain_pending",
                new_callable=AsyncMock,
                return_value=True,
            )
        )
        stack.enter_context(
            patch.object(BudgetManager, "_get_total_spent_sync", return_value=0.0)
        )

        stop = asyncio.Event()
        stop.set()
        return await run_sn_pools(
            cost_limit=1.0,
            scope_run_id=scope_run_id,
            edits_only=True,
            stop_event=stop,
            skip_global_maintenance=policy.bypass_global_maintenance,
        )


def _plant_source_backed_name(gc: Any, sn_id: str) -> str:
    """A live name whose DD source has no upstream edge, as a rebuild leaves it.

    The source reconciliation marks such a source stale and resolves the name
    it produced, so this is exactly a row graph-wide maintenance acts on.
    """
    source_id = f"dd:{sn_id}"
    gc.query(
        """
        MERGE (sn:StandardName {id: $sn_id})
        SET sn.name_stage = 'drafted', sn.origin = 'pipeline',
            sn.validation_status = 'valid', sn.chain_length = 0,
            sn.docs_stage = 'pending', sn.docs_chain_length = 0,
            sn.description = 'Out-of-scope digest probe', sn.kind = 'scalar',
            sn.unit = 'eV'
        MERGE (sns:StandardNameSource {id: $source_id})
        SET sns.source_type = 'dd', sns.source_id = $sn_id,
            sns.status = 'extracted', sns.batch_key = 'scope-probe'
        MERGE (sns)-[:PRODUCED_NAME]->(sn)
        """,
        sn_id=sn_id,
        source_id=source_id,
    )
    return source_id


def _digest(gc: Any, sn_id: str) -> str:
    """A name's stored properties and its one-hop relationships, hashed."""
    node_rows = list(
        gc.query(
            "MATCH (sn:StandardName {id: $id}) RETURN properties(sn) AS props",
            id=sn_id,
        )
    )
    assert node_rows, f"name {sn_id} is absent from the graph"
    rel_rows = list(
        gc.query(
            """
            MATCH (sn:StandardName {id: $id})-[r]-(other)
            RETURN type(r) AS t, startNode(r).id AS s, endNode(r).id AS e,
                   properties(r) AS rp, labels(other) AS ol,
                   properties(other) AS op
            ORDER BY t, s, e
            """,
            id=sn_id,
        )
    )
    payload = {
        "node": node_rows[0]["props"],
        "relationships": [dict(row) for row in rel_rows],
    }
    return hashlib.sha256(
        json.dumps(payload, sort_keys=True, default=str).encode()
    ).hexdigest()


@pytest.mark.graph
@pytest.mark.timeout(120)
def test_scoped_run_leaves_a_planted_out_of_scope_name_unchanged(
    disposable_neo4j_empty: str, request: pytest.FixtureRequest
) -> None:
    from iter_codex.graph.client import GraphClient
    from iter_codex.standard_names.graph_store.graph_audits import (
        reconcile_standard_name_sources,
    )

    token = uuid.uuid4().hex[:8]
    guard_id = f"scope_guard_{token}"
    control_id = f"scope_control_{token}"
    with GraphClient() as gc:
        _plant_source_backed_name(gc, guard_id)
        _plant_source_backed_name(gc, control_id)
        guard_before = _digest(gc, guard_id)
        control_before = _digest(gc, control_id)

    summary = asyncio.run(_run_scoped_pools(f"sn-edit-probe-{token}"))

    # The guard arm: the planted name's properties and relationships are exactly
    # as they were, and the run records which pass it did not run.
    with GraphClient() as gc:
        assert _digest(gc, guard_id) == guard_before

    assert "reconcile_standard_name_sources" in summary.bypassed_maintenance_passes
    with GraphClient() as gc:
        # The control arm: run the same maintenance directly and show it is a
        # pass this planted state would move, so the guard's equality above is
        # an absence of effect and not an inert fixture.
        reconcile_after = reconcile_standard_name_sources("dd")
        assert reconcile_after["stale_marked"] >= 1
        assert _digest(gc, control_id) != control_before


@pytest.mark.graph
@pytest.mark.timeout(120)
def test_focus_stamp_clear_is_bounded_by_the_named_sources(
    disposable_neo4j_empty: str,
) -> None:
    """The one write a focus run makes before seeding stays inside its paths."""
    from iter_codex.cli.sn import FOCUS_STAMP_CLEAR_CYPHER
    from iter_codex.graph.client import GraphClient

    token = uuid.uuid4().hex[:8]
    inside = f"dd:{_FOCUS_PATH}"
    outside = f"dd:equilibrium/time_slice/profiles_1d/psi_outside_{token}"
    with GraphClient() as gc:
        for source_id in (inside, outside):
            gc.query(
                """
                MERGE (sns:StandardNameSource {id: $source_id})
                SET sns.source_type = 'dd', sns.source_id = $source_id,
                    sns.status = 'extracted', sns.run_id = 'previous-run'
                MERGE (sn:StandardName {id: $sn_id})
                SET sn.run_id = 'previous-run', sn.name_stage = 'drafted'
                MERGE (sns)-[:PRODUCED_NAME]->(sn)
                """,
                source_id=source_id,
                sn_id=f"name_for_{source_id}",
            )
        gc.query(FOCUS_STAMP_CLEAR_CYPHER, source_ids=[inside])
        rows = list(
            gc.query(
                """
                MATCH (sns:StandardNameSource)
                OPTIONAL MATCH (sns)-[:PRODUCED_NAME]->(sn:StandardName)
                RETURN sns.id AS source_id, sns.run_id AS source_run,
                       sn.run_id AS name_run
                ORDER BY source_id
                """
            )
        )

    by_source = {row["source_id"]: row for row in rows}
    assert by_source[inside]["source_run"] is None
    assert by_source[inside]["name_run"] is None
    assert by_source[outside]["source_run"] == "previous-run"
    assert by_source[outside]["name_run"] == "previous-run"


@pytest.mark.graph
@pytest.mark.timeout(120)
def test_edits_scope_binds_only_the_pending_successors(
    disposable_neo4j_empty: str,
) -> None:
    """The run scope an edits-only launch mints covers exactly its own names."""
    from iter_codex.cli.sn import _bind_pending_edit_scope
    from iter_codex.graph.client import GraphClient

    token = uuid.uuid4().hex[:8]
    pending_id = f"pending_edit_{token}"
    settled_id = f"settled_edit_{token}"
    with GraphClient() as gc:
        for name_id, edit_status in (
            (pending_id, "open"),
            (settled_id, "applied"),
        ):
            gc.query(
                """
                MERGE (sn:StandardName {id: $id})
                SET sn.name_stage = 'drafted', sn.edit_status = $edit_status,
                    sn.run_id = 'previous-run', sn.validation_status = 'valid'
                """,
                id=name_id,
                edit_status=edit_status,
            )

    scope_run_id = _bind_pending_edit_scope()

    assert scope_run_id, "no scope was minted for the pending successor"
    with GraphClient() as gc:
        rows = {
            row["id"]: row["run_id"]
            for row in gc.query(
                "MATCH (sn:StandardName) WHERE sn.id IN $ids "
                "RETURN sn.id AS id, sn.run_id AS run_id",
                ids=[pending_id, settled_id],
            )
        }
    assert rows[pending_id] == scope_run_id
    assert rows[settled_id] == "previous-run"
