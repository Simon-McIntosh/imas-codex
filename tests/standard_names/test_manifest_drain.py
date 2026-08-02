"""Unit coverage for exact, ephemeral standard-name manifest drains."""

from __future__ import annotations

import asyncio
from contextlib import ExitStack, contextmanager
from pathlib import Path
from unittest.mock import AsyncMock, MagicMock, patch

import pytest
from click.testing import CliRunner

from imas_codex.cli.sn import _compute_pool_progress, _run_sn_cmd, sn_run
from imas_codex.standard_names.graph_ops import (
    MANIFEST_DRAIN_DISPOSITIONS,
    _claim_sn_atomic,
    canonicalize_manifest_drain_paths,
    classify_manifest_drain_item,
    finalize_manifest_drain_scope,
)


def _plan_item(**updates) -> dict:
    item = {
        "path": "equilibrium/time_slice/global_quantities/ip",
        "dd_version": "4.1.0",
        "configured_version_present": True,
        "node": {
            "id": "equilibrium/time_slice/global_quantities/ip",
            "node_category": "quantity",
        },
        "source": None,
        "live_names": [],
        "all_name_ids": [],
        "parents": [],
        "dd_gaps": [],
    }
    item.update(updates)
    return item


def _source(path: str, **updates) -> dict:
    source = {
        "id": f"dd:{path}",
        "source_type": "dd",
        "source_id": path,
        "status": "extracted",
        "dd_version": "4.1.0",
        "dd_snapshot_pinned": True,
        "dd_target_ids": [path],
        "produced_sn_id": None,
        "worker_claim_live": False,
        "scope_conflict": False,
    }
    source.update(updates)
    return source


def test_exact_paths_are_canonical_and_order_preserving() -> None:
    assert canonicalize_manifest_drain_paths(
        ["magnetics/ip", "equilibrium/q", "magnetics/ip"]
    ) == ["magnetics/ip", "equilibrium/q"]


@pytest.mark.parametrize(
    "paths",
    [
        [],
        [""],
        ["dd:magnetics/ip"],
        ["catalog:plasma_current"],
        ["derived:plasma_current"],
        ["signals:tcv:ip"],
        ["magnetics/*"],
        ["magnetics /ip"],
    ],
)
def test_non_exact_path_collections_fail_closed(paths: list[str]) -> None:
    with pytest.raises(ValueError):
        canonicalize_manifest_drain_paths(paths)


def test_five_way_classification_is_exhaustive() -> None:
    path = _plan_item()["path"]
    accepted = {
        "id": "plasma_current",
        "name_stage": "accepted",
        "docs_stage": "accepted",
        "validation_status": "valid",
        "scope_conflict": False,
    }
    cases = [
        _plan_item(
            source=_source(path, produced_sn_id="plasma_current"),
            all_name_ids=["plasma_current"],
            live_names=[accepted],
        ),
        _plan_item(
            source=_source(path, produced_sn_id="plasma_current"),
            all_name_ids=["plasma_current"],
            live_names=[{**accepted, "docs_stage": "drafted"}],
        ),
        _plan_item(),
        _plan_item(node={"id": path, "node_category": "metadata"}),
        _plan_item(configured_version_present=False),
    ]
    assert tuple(
        classify_manifest_drain_item(case)["disposition"] for case in cases
    ) == (
        "accepted",
        "active_in_flight",
        "genuine_gap",
        "non_nameable",
        "ambiguous",
    )
    assert MANIFEST_DRAIN_DISPOSITIONS == (
        "accepted",
        "active_in_flight",
        "genuine_gap",
        "non_nameable",
        "ambiguous",
    )


def test_dd_gap_evidence_never_suppresses_a_genuine_gap() -> None:
    item = _plan_item(
        dd_gaps=[
            {
                "id": "historical-evidence",
                "kind": "vocabulary_gap",
                "status": "open",
                "linked": False,
            }
        ]
    )
    assert classify_manifest_drain_item(item)["disposition"] == "genuine_gap"


def test_active_name_never_becomes_a_gap() -> None:
    path = _plan_item()["path"]
    item = _plan_item(
        source=_source(path, produced_sn_id="plasma_current"),
        all_name_ids=["plasma_current"],
        live_names=[
            {
                "id": "plasma_current",
                "name_stage": "reviewed",
                "docs_stage": "pending",
                "validation_status": "valid",
                "scope_conflict": False,
            }
        ],
    )
    assert classify_manifest_drain_item(item)["disposition"] == "active_in_flight"


def test_scoped_review_restage_is_atomic_in_seed_expand_and_readback(
    monkeypatch,
) -> None:
    transaction = MagicMock()
    transaction.closed = False
    transaction.run.side_effect = [
        [{"_cluster_id": "cluster", "_unit": None, "_physics_domain": None}],
        None,
        [
            {
                "id": "plasma_current",
                "description": "Plasma current.",
                "documentation": None,
                "kind": "scalar",
                "unit": "A",
                "cluster_id": "cluster",
                "physics_domain": "magnetics",
                "validation_status": "valid",
                "claim_token": "token",
                "claim_seq": 1,
            }
        ],
    ]
    session = MagicMock()
    session.begin_transaction.return_value = transaction

    @contextmanager
    def session_context():
        yield session

    client = MagicMock()
    client.__enter__.return_value = client
    client.__exit__.return_value = False
    client.session = session_context
    monkeypatch.setattr(
        "imas_codex.standard_names.graph_ops.GraphClient", lambda: client
    )

    items = _claim_sn_atomic(
        eligibility_where=(
            "sn.name_stage = 'drafted' OR "
            "(sn.name_stage = 'reviewed' "
            "AND sn.reviewer_score_name >= $min_score)"
        ),
        query_params={"min_score": 0.85},
        batch_size=2,
        drain_scope_id="bounded-scope",
        restage_review_axis="name",
    )

    assert [item["id"] for item in items] == ["plasma_current"]
    seed, expand, readback = [call.args[0] for call in transaction.run.call_args_list]
    assert "sn.drain_scope_id = $drain_scope_id" in seed
    assert "sn.drain_scope_id = $drain_scope_id" in expand
    assert "sn.drain_scope_id = $drain_scope_id" in readback
    assert "THEN 'drafted' ELSE sn.name_stage END" in seed
    assert "THEN 'drafted' ELSE sn.name_stage END" in expand
    assert "sn.drain_claim_scope_id = $drain_scope_id" in seed
    assert "sn.drain_claim_scope_id = $drain_scope_id" in expand


def _write_manifest(path: Path) -> None:
    path.write_text(
        "schema_version: 1\nname: bounded-drain\nsources:\n  magnetics:\n    - ip\n",
        encoding="utf-8",
    )


def test_dry_run_stops_before_every_mutator(monkeypatch, tmp_path: Path) -> None:
    manifest = tmp_path / "batch.yaml"
    _write_manifest(manifest)
    plan = [
        classify_manifest_drain_item(
            _plan_item(
                path="magnetics/ip",
                node={"id": "magnetics/ip", "node_category": "quantity"},
            )
        )
    ]
    monkeypatch.setattr("imas_codex.settings.get_dd_version", lambda: "4.1.0")
    monkeypatch.setattr(
        "imas_codex.standard_names.graph_ops.build_manifest_drain_plan",
        lambda paths, **kwargs: plan,
    )
    forbidden = MagicMock(side_effect=AssertionError("dry run crossed write boundary"))
    monkeypatch.setattr(
        "imas_codex.standard_names.graph_ops.claim_manifest_drain_scope", forbidden
    )
    monkeypatch.setattr("imas_codex.cli.sn._require_embed_ready", forbidden)

    result = CliRunner().invoke(sn_run, ["--drain-batch", str(manifest), "--dry-run"])

    assert result.exit_code == 0, result.output
    assert "genuine_gap=1" in result.output
    forbidden.assert_not_called()


def test_live_operator_reports_owned_scope_and_cleans_it(
    monkeypatch, tmp_path: Path
) -> None:
    manifest = tmp_path / "batch.yaml"
    _write_manifest(manifest)
    item = classify_manifest_drain_item(
        _plan_item(
            path="magnetics/ip",
            node={"id": "magnetics/ip", "node_category": "quantity"},
        )
    )
    monkeypatch.setattr("imas_codex.settings.get_dd_version", lambda: "4.1.0")
    embed = MagicMock(side_effect=AssertionError("bounded drain probed embedding"))
    monkeypatch.setattr("imas_codex.cli.sn._require_embed_ready", embed)
    monkeypatch.setattr(
        "imas_codex.standard_names.graph_ops.prepare_manifest_drain_scope",
        lambda paths, **kwargs: ("owned-scope", [item]),
    )
    run = MagicMock(return_value={"drain_report": [item]})
    clear = MagicMock(return_value={"sources": 1, "names": 0})
    monkeypatch.setattr("imas_codex.cli.sn._run_sn_cmd", run)
    monkeypatch.setattr(
        "imas_codex.standard_names.graph_ops.clear_manifest_drain_scope", clear
    )

    result = CliRunner().invoke(sn_run, ["--drain-batch", str(manifest)])

    assert result.exit_code == 0, result.output
    assert "Bounded drain final" in result.output
    assert run.call_args.kwargs["drain_scope_id"] == "owned-scope"
    assert run.call_args.kwargs["skip_global_maintenance"] is True
    embed.assert_not_called()
    clear.assert_called_once_with("owned-scope")


def test_inner_bounded_run_disables_embedding_preflight_and_monitor(
    monkeypatch,
) -> None:
    forbidden = MagicMock(side_effect=AssertionError("embedding preflight called"))
    captured: dict[str, object] = {}

    def run_discovery(config, async_main):
        captured["check_embed"] = config.check_embed
        return {"summary": None}

    monkeypatch.setattr("imas_codex.cli.sn._require_embed_ready", forbidden)
    monkeypatch.setattr("imas_codex.cli.discover.common.use_rich_output", lambda: False)
    monkeypatch.setattr("imas_codex.cli.discover.common.run_discovery", run_discovery)

    assert (
        _run_sn_cmd(
            cost_limit=1.0,
            time_limit=None,
            compose_model=None,
            per_domain_limit=None,
            dry_run=False,
            quiet=True,
            verbose=False,
            drain_scope_id="owned-scope",
            drain_paths=("magnetics/ip",),
            drain_dd_version="4.1.0",
            skip_global_maintenance=True,
        )
        is None
    )
    forbidden.assert_not_called()
    assert captured["check_embed"] is False


def test_pre_loop_cleanup_does_not_mask_the_primary_failure(
    monkeypatch, tmp_path: Path
) -> None:
    manifest = tmp_path / "batch.yaml"
    _write_manifest(manifest)
    item = classify_manifest_drain_item(
        _plan_item(
            path="magnetics/ip",
            node={"id": "magnetics/ip", "node_category": "quantity"},
        )
    )
    monkeypatch.setattr("imas_codex.settings.get_dd_version", lambda: "4.1.0")
    monkeypatch.setattr(
        "imas_codex.standard_names.graph_ops.prepare_manifest_drain_scope",
        lambda paths, **kwargs: ("owned-scope", [item]),
    )
    monkeypatch.setattr(
        "imas_codex.cli.sn._run_sn_cmd",
        MagicMock(side_effect=RuntimeError("primary failure")),
    )
    monkeypatch.setattr(
        "imas_codex.standard_names.graph_ops.clear_manifest_drain_scope",
        MagicMock(side_effect=RuntimeError("cleanup failure")),
    )

    result = CliRunner().invoke(sn_run, ["--drain-batch", str(manifest)])

    assert result.exit_code == 1
    assert isinstance(result.exception, RuntimeError)
    assert str(result.exception) == "primary failure"


@pytest.mark.asyncio
async def test_in_loop_cleanup_does_not_mask_the_primary_failure() -> None:
    from tests.standard_names.test_scoped_global_maintenance import (
        _GO,
        _LOOP,
        _graph_context,
        _maintenance_mocks,
    )

    graph_context, _ = _graph_context()
    with ExitStack() as stack:
        _maintenance_mocks(stack)
        stack.enter_context(patch(f"{_GO}.create_sn_run_open"))
        stack.enter_context(patch(f"{_GO}.finalize_sn_run"))
        stack.enter_context(
            patch(
                "imas_codex.standard_names.ledger.find_provenance_orphans",
                return_value=[],
            )
        )
        stack.enter_context(
            patch(
                "imas_codex.standard_names.audits."
                "find_flux_surface_reduction_violations",
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
                side_effect=RuntimeError("primary failure"),
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
        finalize = stack.enter_context(
            patch(
                f"{_GO}.finalize_manifest_drain_scope",
                side_effect=RuntimeError("cleanup failure"),
            )
        )

        from imas_codex.standard_names.loop import run_sn_pools

        stop = asyncio.Event()
        stop.set()
        with pytest.raises(RuntimeError, match="primary failure"):
            await run_sn_pools(
                cost_limit=5.0,
                drain_scope_id="owned-scope",
                drain_paths=("magnetics/ip",),
                drain_dd_version="4.1.0",
                stop_event=stop,
                skip_global_maintenance=True,
            )

    finalize.assert_called_once()


def test_scope_finalization_preserves_primary_cleanup_failure(monkeypatch) -> None:
    monkeypatch.setattr(
        "imas_codex.standard_names.graph_ops.release_manifest_drain_claims",
        MagicMock(side_effect=RuntimeError("release failure")),
    )
    monkeypatch.setattr(
        "imas_codex.standard_names.graph_ops.clear_manifest_drain_scope",
        MagicMock(side_effect=RuntimeError("clear failure")),
    )

    with pytest.raises(RuntimeError, match="release failure"):
        finalize_manifest_drain_scope(
            "owned-scope",
            paths=["path"],
            dd_version="4.1.0",
        )


def test_scope_finalization_surfaces_clear_failure_after_success(monkeypatch) -> None:
    monkeypatch.setattr(
        "imas_codex.standard_names.graph_ops.release_manifest_drain_claims",
        MagicMock(return_value={"sources": 0, "names": 0}),
    )
    monkeypatch.setattr(
        "imas_codex.standard_names.graph_ops.build_manifest_drain_plan",
        MagicMock(return_value=[]),
    )
    monkeypatch.setattr(
        "imas_codex.standard_names.graph_ops.clear_manifest_drain_scope",
        MagicMock(side_effect=RuntimeError("clear failure")),
    )

    with pytest.raises(RuntimeError, match="clear failure"):
        finalize_manifest_drain_scope("owned-scope", paths=["path"], dd_version="4.1.0")


@pytest.mark.parametrize(
    "source_updates",
    [
        {"status": "failed"},
        {"status": "vocab_gap"},
        {"status": "skipped"},
        {"attempt_count": 5},
        {"produced_sn_id": "exhausted", "dd_target_ids": ["magnetics/ip"]},
    ],
)
def test_terminal_genuine_gaps_are_report_only(source_updates: dict) -> None:
    path = "magnetics/ip"
    item = _plan_item(
        path=path,
        node={"id": path, "node_category": "quantity"},
        source=_source(path, **source_updates),
        all_name_ids=["exhausted"] if source_updates.get("produced_sn_id") else [],
    )
    classified = classify_manifest_drain_item(item)
    assert classified["disposition"] == "genuine_gap"
    assert classified["drain_actionable"] is False


def test_progress_generate_count_requires_actionable_scope() -> None:
    gc = MagicMock()
    gc.query.return_value = [{}]
    _compute_pool_progress(
        gc,
        domains=None,
        rotation_cap=3,
        min_score=0.85,
        drain_scope_id="owned-scope",
    )
    query = gc.query.call_args.args[0]
    pending_generate = query.split("RETURN count(s) AS generate_name", 1)[0]
    assert "s.drain_scope_actionable = true" in pending_generate


@pytest.mark.parametrize(
    "extra",
    [
        ["--focus", "magnetics/ip"],
        ["--source", "signals"],
        ["--reset-to", "drafted"],
        ["--only", "reconcile"],
    ],
)
def test_operator_mode_rejects_competing_selectors(
    monkeypatch, tmp_path: Path, extra: list[str]
) -> None:
    manifest = tmp_path / "batch.yaml"
    _write_manifest(manifest)
    result = CliRunner().invoke(
        sn_run, ["--drain-batch", str(manifest), "--dry-run", *extra]
    )
    assert result.exit_code == 2
    assert "incompatible" in result.output


def test_audit_evidence_is_not_interpreted_as_an_executable_subset(
    tmp_path: Path,
) -> None:
    evidence = tmp_path / "audit.json"
    evidence.write_text(
        '{"schema":"imas-codex.bounded-integrity-manifest",'
        '"sources":[{"source_type":"dd","semantic_id":"magnetics/ip"}]}',
        encoding="utf-8",
    )
    result = CliRunner().invoke(sn_run, ["--drain-batch", str(evidence), "--dry-run"])
    assert result.exit_code == 2
    assert "schema-compliant" in result.output
