"""Tests for inline review after ``sn edit`` staging.

By default ``sn edit`` stages a proposal AND reviews it inline — scoped to
just that edit's ``run_id`` — so the successor lands accepted (or the review
outcome is surfaced) in one command, with no follow-up ``sn run``.  The
``--stage-only`` escape preserves the old stage-and-defer behaviour for bulk /
scripted migrations.

Section 1 — Engine unit tests (``run_inline_review``)
-----------------------------------------------------
``run_sn_pools`` is mocked (AsyncMock) and the outcome-collection graph read
is fed a fake ``GraphClient``.  These assert the scoping (``scope_run_id`` =
the edit's ``run_id``; ``skip_generate`` selects ``--only review`` for
rename/docs and full-pipeline for hint), that the gate is honoured (a
below-threshold successor is reported un-accepted, never force-accepted), and
that a non-applied plan reviews nothing.

Section 2 — CLI wiring
----------------------
``apply_edit`` and ``run_inline_review`` are both mocked; these assert the
default path invokes the inline review and ``--stage-only`` does not.

Section 3 — Graph-marked end-to-end (real Neo4j, auto-skipped when absent)
--------------------------------------------------------------------------
A synthetic ``__inlinetest__``-prefixed drafted successor is created directly
in the graph; ``run_inline_review`` drives the REAL name-review worker (LLM
mocked) scoped to the edit's run_id and lands — or, below threshold, does not
land — the successor.  Deselected by default (``-m "not graph"``).
"""

from __future__ import annotations

import asyncio
import uuid
from types import SimpleNamespace
from unittest.mock import AsyncMock, MagicMock, patch

import pytest
from click.testing import CliRunner

from imas_codex.cli.sn import sn
from imas_codex.standard_names.edit import (
    EditPlan,
    InlineReviewOutcome,
    InlineReviewResult,
    run_inline_review,
)

_RUN_SN_POOLS = "imas_codex.standard_names.loop.run_sn_pools"


def _rename_plan(**overrides) -> EditPlan:
    base = {
        "target": "electron_temperature",
        "mode": "rename",
        "axis": "name",
        "scope": "only_self",
        "entry": "review_name",
        "successor": "ion_temperature",
        "cascade_planned": [],
        "blocked": None,
        "actions": ["renamed 'electron_temperature' → 'ion_temperature'"],
        "applied": True,
        "run_id": "sn-edit-20260715T101010Z",
    }
    base.update(overrides)
    return EditPlan(**base)


def _fake_gc(rows: list[dict]) -> MagicMock:
    gc = MagicMock()
    gc.query = MagicMock(return_value=rows)
    return gc


def _summary(cost: float = 0.021, stop: str = "no_eligible_work") -> SimpleNamespace:
    return SimpleNamespace(cost_spent=cost, stop_reason=stop)


# ---------------------------------------------------------------------------
# Section 1 — run_inline_review scoping + gate
# ---------------------------------------------------------------------------


class TestInlineReviewScoping:
    def test_rename_scopes_to_run_id_and_reviews_only(self):
        """A rename reviews --only (skip_generate) scoped to the edit's run_id."""
        plan = _rename_plan()
        rows = [
            {
                "id": "ion_temperature",
                "name_stage": "accepted",
                "docs_stage": "pending",
                "edit_status": "applied",
                "reviewer_score_name": 0.86,
                "reviewer_score_docs": None,
            }
        ]
        mock_pools = AsyncMock(return_value=_summary())
        with patch(_RUN_SN_POOLS, mock_pools):
            outcome = run_inline_review(plan, cost_limit=0.5, gc=_fake_gc(rows))

        assert mock_pools.await_count == 1
        kwargs = mock_pools.await_args.kwargs
        assert kwargs["scope_run_id"] == plan.run_id
        assert kwargs["skip_generate"] is True  # --only review
        assert kwargs["cost_limit"] == 0.5

        assert outcome.ran is True
        assert outcome.cost == pytest.approx(0.021)
        assert outcome.stop_reason == "no_eligible_work"
        assert len(outcome.results) == 1
        r = outcome.results[0]
        assert r.id == "ion_temperature"
        assert r.accepted is True
        assert r.name_stage == "accepted"
        assert r.edit_status == "applied"
        assert outcome.all_accepted is True

    def test_hint_keeps_generate_pool(self):
        """A hint edit regenerates, so the generate pool must NOT be skipped."""
        plan = _rename_plan(mode="hint", axis="name", entry="generate", successor=None)
        rows = [
            {
                "id": "electron_temperature",
                "name_stage": "accepted",
                "docs_stage": "pending",
                "edit_status": "applied",
                "reviewer_score_name": 0.9,
                "reviewer_score_docs": None,
            }
        ]
        mock_pools = AsyncMock(return_value=_summary())
        with patch(_RUN_SN_POOLS, mock_pools):
            outcome = run_inline_review(plan, cost_limit=1.0, gc=_fake_gc(rows))

        assert mock_pools.await_args.kwargs["skip_generate"] is False
        # hint settles the target in place, not a successor
        assert [r.id for r in outcome.results] == ["electron_temperature"]

    def test_cascade_successors_are_reported(self):
        """A family/subtree rename reports the successor + cascade descendants."""
        plan = _rename_plan(
            scope="family",
            cascade_planned=[
                {"from": "electron_temperature_a", "to": "ion_temperature_a"}
            ],
        )
        rows = [
            {
                "id": "ion_temperature",
                "name_stage": "accepted",
                "docs_stage": "pending",
                "edit_status": "applied",
                "reviewer_score_name": 0.86,
                "reviewer_score_docs": None,
            },
            {
                "id": "ion_temperature_a",
                "name_stage": "accepted",
                "docs_stage": "pending",
                "edit_status": "applied",
                "reviewer_score_name": None,
                "reviewer_score_docs": None,
            },
        ]
        with patch(_RUN_SN_POOLS, AsyncMock(return_value=_summary())):
            outcome = run_inline_review(plan, cost_limit=1.0, gc=_fake_gc(rows))

        assert {r.id for r in outcome.results} == {
            "ion_temperature",
            "ion_temperature_a",
        }
        assert outcome.run_id == plan.run_id
        assert outcome.all_accepted is True

    def test_below_threshold_not_force_accepted(self):
        """A below-threshold review leaves the successor un-accepted (gate held)."""
        plan = _rename_plan()
        rows = [
            {
                "id": "ion_temperature",
                "name_stage": "reviewed",
                "docs_stage": "pending",
                "edit_status": "open",
                "reviewer_score_name": 0.52,
                "reviewer_score_docs": None,
            }
        ]
        with patch(_RUN_SN_POOLS, AsyncMock(return_value=_summary())):
            outcome = run_inline_review(plan, cost_limit=1.0, gc=_fake_gc(rows))

        r = outcome.results[0]
        assert r.accepted is False
        assert r.name_stage == "reviewed"
        assert r.edit_status == "open"
        assert r.reviewer_score_name == pytest.approx(0.52)
        assert outcome.all_accepted is False
        assert outcome.ran is True

    def test_exhausted_reported_not_accepted(self):
        plan = _rename_plan()
        rows = [
            {
                "id": "ion_temperature",
                "name_stage": "exhausted",
                "docs_stage": "pending",
                "edit_status": "exhausted",
                "reviewer_score_name": 0.58,
                "reviewer_score_docs": None,
            }
        ]
        with patch(_RUN_SN_POOLS, AsyncMock(return_value=_summary())):
            outcome = run_inline_review(plan, cost_limit=1.0, gc=_fake_gc(rows))
        assert outcome.results[0].accepted is False
        assert outcome.results[0].name_stage == "exhausted"
        assert outcome.all_accepted is False

    def test_docs_axis_accepts_on_docs_stage(self):
        """A docs edit accepts on docs_stage, not name_stage."""
        plan = _rename_plan(
            mode="docs", axis="docs", entry="review_docs", successor=None
        )
        rows = [
            {
                "id": "electron_temperature",
                "name_stage": "accepted",
                "docs_stage": "accepted",
                "edit_status": "applied",
                "reviewer_score_name": 0.9,
                "reviewer_score_docs": 0.82,
            }
        ]
        mock_pools = AsyncMock(return_value=_summary())
        with patch(_RUN_SN_POOLS, mock_pools):
            outcome = run_inline_review(plan, cost_limit=1.0, gc=_fake_gc(rows))
        assert mock_pools.await_args.kwargs["skip_generate"] is True
        assert outcome.results[0].accepted is True

    def test_not_applied_plan_reviews_nothing(self):
        """A blocked / dry-run plan (no run_id) does not invoke the pipeline."""
        plan = _rename_plan(applied=False, run_id=None, successor=None)
        mock_pools = AsyncMock(return_value=_summary())
        with patch(_RUN_SN_POOLS, mock_pools):
            outcome = run_inline_review(plan, cost_limit=1.0, gc=_fake_gc([]))
        assert mock_pools.await_count == 0
        assert outcome.ran is False
        assert outcome.results == []
        assert outcome.all_accepted is False

    def test_blocked_plan_reviews_nothing(self):
        plan = _rename_plan(applied=True, blocked="cascade conflict", run_id=None)
        mock_pools = AsyncMock(return_value=_summary())
        with patch(_RUN_SN_POOLS, mock_pools):
            outcome = run_inline_review(plan, cost_limit=1.0, gc=_fake_gc([]))
        assert mock_pools.await_count == 0
        assert outcome.ran is False


# ---------------------------------------------------------------------------
# Section 2 — CLI wiring: default reviews inline, --stage-only defers
# ---------------------------------------------------------------------------


def _accepted_outcome() -> InlineReviewOutcome:
    return InlineReviewOutcome(
        ran=True,
        run_id="sn-edit-20260715T101010Z",
        cost=0.02,
        stop_reason="no_eligible_work",
        results=[
            InlineReviewResult(
                id="ion_temperature",
                name_stage="accepted",
                docs_stage="pending",
                edit_status="applied",
                reviewer_score_name=0.86,
                reviewer_score_docs=None,
                accepted=True,
            )
        ],
    )


class TestCliInlineWiring:
    def test_default_invokes_inline_review(self):
        runner = CliRunner()
        with (
            patch(
                "imas_codex.standard_names.edit.apply_edit",
                return_value=_rename_plan(),
            ),
            patch(
                "imas_codex.standard_names.edit.run_inline_review",
                return_value=_accepted_outcome(),
            ) as mock_review,
            patch("imas_codex.cli.sn._require_embed_ready"),
        ):
            result = runner.invoke(
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
        mock_review.assert_called_once()
        # the just-staged plan (carrying the edit run_id) is what we review
        passed_plan = mock_review.call_args.args[0]
        assert passed_plan.run_id == "sn-edit-20260715T101010Z"
        assert mock_review.call_args.kwargs["cost_limit"] == 1.0
        assert "accepted" in result.output

    def test_cost_limit_passthrough(self):
        runner = CliRunner()
        with (
            patch(
                "imas_codex.standard_names.edit.apply_edit",
                return_value=_rename_plan(),
            ),
            patch(
                "imas_codex.standard_names.edit.run_inline_review",
                return_value=_accepted_outcome(),
            ) as mock_review,
            patch("imas_codex.cli.sn._require_embed_ready"),
        ):
            result = runner.invoke(
                sn,
                [
                    "edit",
                    "electron_temperature",
                    "--rename",
                    "ion_temperature",
                    "--reason",
                    "because",
                    "-c",
                    "3.5",
                ],
            )
        assert result.exit_code == 0, result.output
        assert mock_review.call_args.kwargs["cost_limit"] == 3.5

    def test_stage_only_skips_inline_review(self):
        runner = CliRunner()
        with (
            patch(
                "imas_codex.standard_names.edit.apply_edit",
                return_value=_rename_plan(),
            ),
            patch(
                "imas_codex.standard_names.edit.run_inline_review",
            ) as mock_review,
            patch("imas_codex.cli.sn._require_embed_ready") as mock_embed,
        ):
            result = runner.invoke(
                sn,
                [
                    "edit",
                    "electron_temperature",
                    "--rename",
                    "ion_temperature",
                    "--reason",
                    "because",
                    "--stage-only",
                ],
            )
        assert result.exit_code == 0, result.output
        mock_review.assert_not_called()
        mock_embed.assert_not_called()
        # stage-only keeps the "claimed by the next review rotation" hint
        assert "sn run" in result.output

    def test_no_review_alias(self):
        runner = CliRunner()
        with (
            patch(
                "imas_codex.standard_names.edit.apply_edit",
                return_value=_rename_plan(),
            ),
            patch("imas_codex.standard_names.edit.run_inline_review") as mock_review,
            patch("imas_codex.cli.sn._require_embed_ready"),
        ):
            result = runner.invoke(
                sn,
                [
                    "edit",
                    "electron_temperature",
                    "--rename",
                    "ion_temperature",
                    "--reason",
                    "because",
                    "--no-review",
                ],
            )
        assert result.exit_code == 0, result.output
        mock_review.assert_not_called()

    def test_failed_inline_review_exits_3(self):
        """A below-threshold inline review is a signal: distinct non-zero exit."""
        failed = InlineReviewOutcome(
            ran=True,
            run_id="sn-edit-20260715T101010Z",
            cost=0.03,
            stop_reason="no_eligible_work",
            results=[
                InlineReviewResult(
                    id="ion_temperature",
                    name_stage="reviewed",
                    docs_stage="pending",
                    edit_status="open",
                    reviewer_score_name=0.52,
                    reviewer_score_docs=None,
                    accepted=False,
                )
            ],
        )
        runner = CliRunner()
        with (
            patch(
                "imas_codex.standard_names.edit.apply_edit",
                return_value=_rename_plan(),
            ),
            patch(
                "imas_codex.standard_names.edit.run_inline_review",
                return_value=failed,
            ),
            patch("imas_codex.cli.sn._require_embed_ready"),
        ):
            result = runner.invoke(
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
        assert result.exit_code == 3, result.output
        assert "below threshold" in result.output

    def test_dry_run_does_not_review(self):
        plan = _rename_plan(applied=False, successor=None)
        runner = CliRunner()
        with (
            patch("imas_codex.standard_names.edit.apply_edit", return_value=plan),
            patch("imas_codex.standard_names.edit.run_inline_review") as mock_review,
        ):
            result = runner.invoke(
                sn,
                [
                    "edit",
                    "electron_temperature",
                    "--rename",
                    "ion_temperature",
                    "--reason",
                    "because",
                    "--dry-run",
                ],
            )
        assert result.exit_code == 0, result.output
        mock_review.assert_not_called()

    def test_blocked_exits_2_before_review(self):
        plan = _rename_plan(blocked="target not found", applied=False, run_id=None)
        runner = CliRunner()
        with (
            patch("imas_codex.standard_names.edit.apply_edit", return_value=plan),
            patch("imas_codex.standard_names.edit.run_inline_review") as mock_review,
        ):
            result = runner.invoke(
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
        assert result.exit_code == 2
        mock_review.assert_not_called()

    def test_help_lists_new_flags(self):
        runner = CliRunner()
        result = runner.invoke(sn, ["edit", "--help"])
        assert result.exit_code == 0
        assert "--stage-only" in result.output
        assert "--cost-limit" in result.output or "-c," in result.output


# ---------------------------------------------------------------------------
# Section 3 — Graph-marked end-to-end (real Neo4j, auto-skipped when absent)
# ---------------------------------------------------------------------------

_TEST_ID_PREFIX = "__inlinetest__"


@pytest.fixture()
def _gc():
    try:
        from imas_codex.graph.client import GraphClient

        client = GraphClient()
        client.get_stats()
    except Exception as exc:  # noqa: BLE001
        pytest.skip(f"Neo4j not available: {exc}")
    yield client
    client.close()


@pytest.fixture()
def _clean_inline_nodes(_gc):
    def _wipe():
        _gc.query(
            "MATCH (n) WHERE n.id STARTS WITH $prefix DETACH DELETE n",
            prefix=_TEST_ID_PREFIX,
        )

    _wipe()
    yield
    _wipe()


def _create_drafted_edit_successor(gc, sn_id: str, run_id: str, domain: str) -> None:
    """A drafted rename successor exactly as ``apply_edit`` leaves one.

    The id carries the ``__inlinetest__`` prefix so it can never match a real
    name; the isolated synthetic ``physics_domain`` keeps the review claim off
    any live node.
    """
    gc.query(
        """
        MERGE (sn:StandardName {id: $id})
        SET sn.name             = $id,
            sn.description       = 'synthetic inline-review successor',
            sn.name_stage        = 'drafted',
            sn.docs_stage        = 'pending',
            sn.chain_length      = 0,
            sn.docs_chain_length = 0,
            sn.kind              = 'scalar',
            sn.unit              = 'eV',
            sn.physics_domain    = $domain,
            sn.validation_status = 'valid',
            sn.edit_mode         = 'rename',
            sn.edit_status       = 'open',
            sn.edit_origin       = 'human',
            sn.edit_reason       = 'inline review e2e test',
            sn.run_id            = $run_id,
            sn.claim_token       = NULL,
            sn.claimed_at        = NULL
        """,
        id=sn_id,
        run_id=run_id,
        domain=domain,
    )


def _make_review_item(sn_id: str, token: str, domain: str) -> dict:
    return {
        "id": sn_id,
        "description": "synthetic inline-review successor",
        "documentation": None,
        "kind": "scalar",
        "unit": "eV",
        "cluster_id": None,
        "physics_domain": [domain],
        "validation_status": "valid",
        "claim_token": token,
        "name": sn_id,
        "chain_length": 0,
        "name_stage": "drafted",
    }


def _fake_scoped_pool(_gc, mock_llm, *, accept: bool):
    """An async ``run_sn_pools`` stand-in that runs the REAL review worker.

    Claims the scoped successor (proving run_id scoping restricts the claim to
    this edit) and reviews it with the real name-review worker under a mocked
    LLM verdict, so the drafted → accepted / reviewed transition is exercised
    on the live graph — the substance of an inline review, minus the pool
    harness the unit tests already cover.
    """
    from imas_codex.standard_names.graph_ops import claim_review_name_batch
    from imas_codex.standard_names.models import (
        StandardNameQualityReviewNameOnly,
        StandardNameQualityScoreNameOnly,
    )
    from imas_codex.standard_names.workers import process_review_name_batch

    if accept:
        scores = StandardNameQualityScoreNameOnly(
            grammar=17, semantic=17, convention=17, completeness=17
        )  # 68/80 = 0.85
    else:
        scores = StandardNameQualityScoreNameOnly(
            grammar=11, semantic=11, convention=11, completeness=11
        )  # 44/80 = 0.55

    async def _run(**kwargs):
        run_id = kwargs["scope_run_id"]
        claimed = claim_review_name_batch(scope_run_id=run_id, batch_size=10)
        assert claimed, "scoped claim found no successor"
        # scope must restrict to exactly this edit's stamped node(s)
        for item in claimed:
            resp = StandardNameQualityReviewNameOnly(
                source_id=item["id"],
                standard_name=item["id"],
                scores=scores,
                reasoning="inline review e2e",
            )
            mock_llm.add_response("review_name", response=resp)
            mock_llm.add_response("review_name", response=resp)
        mgr = MagicMock()
        lease = MagicMock()
        lease.charge_event = MagicMock(return_value=SimpleNamespace(overspend=0.0))
        lease.release_unused = MagicMock(return_value=0.0)
        mgr.reserve = MagicMock(return_value=lease)
        n = await process_review_name_batch(
            [dict(c) for c in claimed], mgr, asyncio.Event()
        )
        assert n == len(claimed)
        return SimpleNamespace(cost_spent=0.0, stop_reason="no_eligible_work")

    return _run


@pytest.mark.graph
@pytest.mark.integration
def test_inline_review_lands_successor_accepted(_gc, _clean_inline_nodes, mock_llm):
    run_id = f"sn-edit-inlinetest-{uuid.uuid4().hex[:8]}"
    sn_id = f"{_TEST_ID_PREFIX}accept_{uuid.uuid4().hex[:8]}"
    domain = f"{_TEST_ID_PREFIX}domain_{uuid.uuid4().hex[:6]}"
    _create_drafted_edit_successor(_gc, sn_id, run_id, domain)

    plan = EditPlan(
        target="__inlinetest__original",
        mode="rename",
        axis="name",
        scope="only_self",
        entry="review_name",
        successor=sn_id,
        cascade_planned=[],
        blocked=None,
        actions=[],
        applied=True,
        run_id=run_id,
    )

    with patch(_RUN_SN_POOLS, _fake_scoped_pool(_gc, mock_llm, accept=True)):
        outcome = run_inline_review(plan, cost_limit=5.0, gc=_gc)

    row = _gc.query(
        "MATCH (sn:StandardName {id: $id}) "
        "RETURN sn.name_stage AS name_stage, sn.edit_status AS edit_status",
        id=sn_id,
    )[0]
    assert row["name_stage"] == "accepted", row
    assert row["edit_status"] == "applied", row
    assert outcome.all_accepted is True
    assert outcome.results[0].accepted is True


@pytest.mark.graph
@pytest.mark.integration
def test_inline_review_below_threshold_stays_unaccepted(
    _gc, _clean_inline_nodes, mock_llm
):
    """A failed review is honoured, not force-accepted — the gate holds."""
    run_id = f"sn-edit-inlinetest-{uuid.uuid4().hex[:8]}"
    sn_id = f"{_TEST_ID_PREFIX}reject_{uuid.uuid4().hex[:8]}"
    domain = f"{_TEST_ID_PREFIX}domain_{uuid.uuid4().hex[:6]}"
    _create_drafted_edit_successor(_gc, sn_id, run_id, domain)

    plan = EditPlan(
        target="__inlinetest__original",
        mode="rename",
        axis="name",
        scope="only_self",
        entry="review_name",
        successor=sn_id,
        cascade_planned=[],
        blocked=None,
        actions=[],
        applied=True,
        run_id=run_id,
    )

    with patch(_RUN_SN_POOLS, _fake_scoped_pool(_gc, mock_llm, accept=False)):
        outcome = run_inline_review(plan, cost_limit=5.0, gc=_gc)

    row = _gc.query(
        "MATCH (sn:StandardName {id: $id}) RETURN sn.name_stage AS name_stage",
        id=sn_id,
    )[0]
    assert row["name_stage"] != "accepted", row
    assert outcome.all_accepted is False
    assert outcome.results[0].accepted is False


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
