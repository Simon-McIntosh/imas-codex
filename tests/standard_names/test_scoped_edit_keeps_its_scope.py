"""A scoped ``sn edit`` stays scoped: it does not escalate into a global pass.

``sn edit`` stages one identity and reviews exactly that identity, so its launch
carries a ``scope_run_id`` by construction.  ``run_sn_pools`` runs graph-wide
startup, background and post-drain maintenance unless ``skip_global_maintenance``
is set, and its own default is ``False`` — so whether a single-identity edit
escalates into sourceless-name, attachment-consistency, source-ledger and
derived-parent reconciliation is decided at the edit path's launch, not by the
loop.  An operator running a fenced edit therefore used to get a graph-wide pass
they did not ask for, which refuses on protected identities outside their fence.

These tests pin that launch at the boundary where ``run_sn_pools`` is called,
from the CLI in (the entry point an operator uses) and from the shared pipeline
frame (which the rescore caller also passes through).  Each asserts two
known-present controls — ``scope_run_id`` and ``cost_limit`` — before the
bypass, so a test whose recorder never reached the seam fails on the control
rather than passing on an absent keyword.
"""

from __future__ import annotations

from types import SimpleNamespace
from unittest.mock import AsyncMock, MagicMock, patch

from click.testing import CliRunner

from imas_codex.cli.sn import sn
from imas_codex.standard_names.edit import EditPlan, InlineReviewResult

_RUN_SN_POOLS = "imas_codex.standard_names.loop.run_sn_pools"
_EDIT = "imas_codex.standard_names.edit"


def _rename_plan() -> EditPlan:
    return EditPlan(
        target="electron_temperature",
        mode="rename",
        axis="name",
        scope="only_self",
        entry="review_name",
        successor="ion_temperature",
        cascade_deferred=[],
        blocked=None,
        actions=["renamed 'electron_temperature' → 'ion_temperature'"],
        applied=True,
        run_id="sn-edit-scope-probe",
    )


def _summary() -> SimpleNamespace:
    return SimpleNamespace(cost_spent=0.021, stop_reason="no_eligible_work")


def _landed_successor() -> list[InlineReviewResult]:
    """The successor the stubbed review reports, so the CLI exits on a landing."""
    return [
        InlineReviewResult(
            id="ion_temperature",
            name_stage="accepted",
            docs_stage="pending",
            edit_status="applied",
            reviewer_score_name=0.86,
            reviewer_score_docs=None,
            accepted=True,
        )
    ]


def test_sn_edit_launch_reaches_the_pool_orchestrator_without_global_maintenance():
    """The CLI an operator runs lands on the pool orchestrator carrying the bypass."""
    plan = _rename_plan()
    pools = AsyncMock(return_value=_summary())

    with (
        patch(f"{_EDIT}.apply_edit", return_value=plan),
        patch(f"{_EDIT}._collect_inline_outcomes", return_value=_landed_successor()),
        patch(f"{_EDIT}.GraphClient", MagicMock()),
        patch("imas_codex.cli.sn._require_embed_ready"),
        patch(_RUN_SN_POOLS, pools),
    ):
        result = CliRunner().invoke(
            sn,
            [
                "edit",
                "electron_temperature",
                "--rename",
                "ion_temperature",
                "--reason",
                "because",
            ],
        )

    assert result.exit_code == 0, result.output
    pools.assert_called_once()
    kwargs = pools.call_args.kwargs
    # known-present controls: the recorder saw the seam this test is about.
    assert kwargs["scope_run_id"] == plan.run_id
    assert kwargs["cost_limit"] == 1.0
    assert kwargs.get("skip_global_maintenance") is True


def test_scoped_pipeline_pairs_the_bypass_with_the_scope_it_was_given():
    """The shared frame pairs the bypass with its scope, so both callers inherit it."""
    from imas_codex.standard_names.edit import _run_scoped_pipeline

    pools = AsyncMock(return_value=_summary())
    with patch(_RUN_SN_POOLS, pools):
        _run_scoped_pipeline(
            run_id="sn-edit-scope-probe",
            skip_generate=True,
            cost_limit=2.5,
            min_score=None,
            rotation_cap=None,
            pending_fn=None,
        )

    pools.assert_called_once()
    kwargs = pools.call_args.kwargs
    assert kwargs["scope_run_id"] == "sn-edit-scope-probe"
    assert kwargs["cost_limit"] == 2.5
    assert kwargs.get("skip_global_maintenance") is True
