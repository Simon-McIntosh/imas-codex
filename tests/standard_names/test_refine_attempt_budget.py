"""The name-refinement budget is charged per attempt and is terminal.

``chain_length`` counts persisted successors. A refine attempt that never
persists one — the proposed identity is already taken, the persistence fence
refuses it, the candidate is ungrammatical — leaves it untouched, so a
predicate reading lineage depth re-selects the same name on every poll and
re-bills it. These tests pin the counter that closes that loop: it is charged
on the claim, it is inherited by a successor, it drives escalation and
exhaustion, and a decided conflict spends none of what remains.
"""

from __future__ import annotations

import asyncio
from contextlib import contextmanager
from typing import Any
from unittest.mock import MagicMock, patch

import pytest

_GC_PATH = "imas_codex.standard_names.graph_ops.GraphClient"
_GC_WORKERS_PATH = "imas_codex.graph.client.GraphClient"


def _mock_gc(rows: list[dict[str, Any]] | None = None):
    gc = MagicMock()
    gc.__enter__ = MagicMock(return_value=gc)
    gc.__exit__ = MagicMock(return_value=False)
    gc.query = MagicMock(return_value=rows if rows is not None else [])
    return gc


@contextmanager
def _patch_gc(gc):
    with patch(_GC_PATH, return_value=gc):
        yield


def _mock_budget_manager():
    mgr = MagicMock()
    lease = MagicMock()
    lease.charge_event = MagicMock()
    mgr.reserve = MagicMock(return_value=lease)
    return mgr


def _refine_item(**overrides: Any) -> dict[str, Any]:
    item: dict[str, Any] = {
        "id": "test_name",
        "description": "A test quantity",
        "documentation": None,
        "kind": "scalar",
        "unit": "eV",
        "physics_domain": ["core_profiles"],
        "validation_status": "valid",
        "reviewer_score_name": 0.6,
        "reviewer_comments_per_dim_name": None,
        "chain_length": 0,
        "refine_attempts": 1,
        "name_stage": "refining",
        "source_paths": ["core_profiles/profiles_1d/electrons/temperature"],
        "claim_token": "tok-abc-123",
        "chain_history": [],
    }
    item.update(overrides)
    return item


class TestBudgetIsChargedOnTheClaim:
    """Every verified claim spends a rotation before any model call."""

    def test_charge_is_fenced_on_the_claim_and_returns_the_attempt_number(self):
        from imas_codex.standard_names.graph_ops import _charge_refine_name_attempts

        gc = _mock_gc([{"id": "a", "refine_attempts": 2}])
        with _patch_gc(gc):
            items = _charge_refine_name_attempts(
                [{"id": "a", "claim_token": "tok"}],
            )

        assert items == [{"id": "a", "claim_token": "tok", "refine_attempts": 2}]
        cypher = " ".join(gc.query.call_args.args[0].split())
        assert "sn.claim_token = $token" in cypher
        assert "sn.name_stage = 'refining'" in cypher
        # A name minted before the counter existed starts from its lineage
        # depth, so the fallback never hands back rotations already spent.
        assert (
            "coalesce( sn.refine_attempts, coalesce(sn.chain_length, 0) ) + 1" in cypher
        )

    def test_uncharged_claim_is_dropped_before_the_model_call(self):
        from imas_codex.standard_names.graph_ops import _charge_refine_name_attempts

        gc = _mock_gc([])
        with _patch_gc(gc):
            items = _charge_refine_name_attempts([{"id": "a", "claim_token": "tok"}])

        assert items == []

    def test_one_query_per_claim_token(self):
        from imas_codex.standard_names.graph_ops import _charge_refine_name_attempts

        gc = _mock_gc(
            [
                {"id": "a", "refine_attempts": 1},
                {"id": "b", "refine_attempts": 1},
            ]
        )
        with _patch_gc(gc):
            _charge_refine_name_attempts(
                [
                    {"id": "a", "claim_token": "tok"},
                    {"id": "b", "claim_token": "tok"},
                ]
            )

        assert gc.query.call_count == 1
        assert gc.query.call_args.kwargs["sn_ids"] == ["a", "b"]


class TestEligibilityReadsTheBudget:
    """The claim predicate gates on rotations spent, not on lineage depth."""

    def test_predicate_gates_on_the_attempt_counter(self):
        from imas_codex.standard_names.graph_ops import (
            REFINE_NAME_ATTEMPTS_SPENT,
            REFINE_NAME_ELIGIBILITY_WHERE,
        )

        assert f"{REFINE_NAME_ATTEMPTS_SPENT} < $rotation_cap" in (
            REFINE_NAME_ELIGIBILITY_WHERE
        )
        # Lineage depth may still be projected, but it must not be the gate.
        assert "coalesce(sn.chain_length, 0) < $rotation_cap" not in (
            REFINE_NAME_ELIGIBILITY_WHERE
        )


class TestBudgetFollowsTheLineage:
    """A successor inherits the rotations its predecessor spent."""

    def test_successor_inherits_the_predecessor_count(self):
        from imas_codex.standard_names.graph_ops import (
            RefinedNamePersistenceRefusal,
            persist_refined_name,
        )

        gc = MagicMock()
        gc.__enter__ = MagicMock(return_value=gc)
        gc.__exit__ = MagicMock(return_value=False)
        tx = MagicMock()
        tx.closed = False
        tx.run.return_value = []
        session = MagicMock()
        session.begin_transaction = MagicMock(return_value=tx)

        @contextmanager
        def _session_ctx():
            yield session

        gc.session = _session_ctx

        # The preflight cannot commit against a mocked predecessor; the query
        # it issued is what this test reads.
        with _patch_gc(gc), pytest.raises(RefinedNamePersistenceRefusal):
            persist_refined_name(
                old_name="old",
                new_name="new",
                description="d",
                old_chain_length=1,
            )

        cypher = " ".join(tx.run.call_args_list[0].args[0].split())
        assert "new.refine_attempts = coalesce( old.refine_attempts" in cypher


class TestExhaustionIsDecidedByTheBudget:
    """Review parks a below-threshold name once its rotations are gone."""

    @pytest.mark.parametrize(
        ("attempts", "expected"),
        [(0, "reviewed"), (2, "reviewed"), (3, "exhausted"), (4, "exhausted")],
    )
    def test_stage_follows_the_rotations_spent(self, attempts: int, expected: str):
        from imas_codex.standard_names.graph_ops import persist_reviewed_name

        read_row = {
            "chain_length": 0,
            "refine_attempts": attempts,
            "validation_status": "valid",
            "edit_status": None,
            "edit_scope": None,
            "edit_override_edits": False,
            "edit_include_accepted": False,
        }
        gc = MagicMock()
        gc.__enter__ = MagicMock(return_value=gc)
        gc.__exit__ = MagicMock(return_value=False)
        gc.query = MagicMock(side_effect=[[read_row], [{"id": "electron_temperature"}]])

        with _patch_gc(gc):
            stage = persist_reviewed_name(
                sn_id="electron_temperature",
                claim_token="tok",
                score=0.5,
                model="m",
                rotation_cap=3,
                skip_review_node=True,
                resolution_method="quorum_consensus",
                reviewer_chain_size=2,
            )

        assert stage == expected
        # A name with zero lineage depth is exhausted purely on rotations,
        # which is the case a chain-depth test could never reach.
        assert gc.query.call_args.kwargs["target_stage"] == expected


class TestDecidedConflictsDoNotSpendTheRest:
    """A conflict no further attempt can resolve parks the name at once."""

    @pytest.mark.parametrize(
        "reason",
        [
            "successor_collision",
            "successor_lifecycle_collision",
            "vocabulary_gap",
            "grammar_invalid",
        ],
    )
    def test_terminal_reasons_park_regardless_of_remaining_budget(self, reason: str):
        from imas_codex.standard_names.graph_ops import stop_refine_name_attempt

        gc = _mock_gc([{"stage": "exhausted"}])
        with _patch_gc(gc):
            stop_refine_name_attempt(sn_id="a", token="tok", reason=reason)

        assert gc.query.call_args.kwargs["terminal"] is True

    def test_transient_failure_keeps_the_remaining_budget(self):
        from imas_codex.standard_names.graph_ops import stop_refine_name_attempt

        gc = _mock_gc([{"stage": "reviewed"}])
        with _patch_gc(gc):
            stop_refine_name_attempt(sn_id="a", token="tok", reason="transient_failure")

        assert gc.query.call_args.kwargs["terminal"] is False

    def test_the_reason_is_queryable_and_names_the_occupied_identity(self):
        from imas_codex.standard_names.graph_ops import stop_refine_name_attempt

        gc = _mock_gc([{"stage": "exhausted"}])
        with _patch_gc(gc):
            stop_refine_name_attempt(
                sn_id="a",
                token="tok",
                reason="successor_collision",
                collision_name="b",
            )

        cypher = " ".join(gc.query.call_args.args[0].split())
        assert "sn.refine_stop_reason =" in cypher
        assert "sn.refine_stopped_at = datetime()" in cypher
        assert "sn.refine_collision_name = $collision_name" in cypher
        assert gc.query.call_args.kwargs["collision_name"] == "b"

    def test_spent_budget_parks_with_its_own_reason(self):
        """A stop that is not itself decided still parks at the cap."""
        from imas_codex.standard_names.graph_ops import stop_refine_name_attempt

        gc = _mock_gc([{"stage": "exhausted"}])
        with _patch_gc(gc):
            stop_refine_name_attempt(
                sn_id="a", token="tok", reason="transient_failure", rotation_cap=3
            )

        cypher = " ".join(gc.query.call_args.args[0].split())
        # The reason recorded at the cap is the spent budget, not the last
        # symptom, so a parked name can be enumerated by cause.
        assert (
            "WHEN target_stage = 'exhausted' AND NOT $terminal THEN $attempts_exhausted"
            in cypher
        )
        assert gc.query.call_args.kwargs["attempts_exhausted"] == "attempts_exhausted"


class TestEscalationFiresOnTheLastRotation:
    """The final rotation runs on the escalation seat, persisted or not."""

    @pytest.mark.asyncio
    @pytest.mark.parametrize(
        ("attempts", "escalates"), [(1, False), (2, False), (3, True)]
    )
    async def test_last_rotation_uses_the_escalation_seat(
        self, attempts: int, escalates: bool
    ):
        from imas_codex.standard_names.defaults import DEFAULT_ESCALATION_MODEL
        from imas_codex.standard_names.workers import process_refine_name_batch

        # chain_length stays 0 throughout: nothing this name proposed has ever
        # persisted, which is exactly when the lineage-depth test never fired.
        item = _refine_item(chain_length=0, refine_attempts=attempts)
        seen: dict[str, Any] = {}

        async def _llm(**kwargs):
            seen["model"] = kwargs["model"]
            raise RuntimeError("stop here — the model choice is what matters")

        with (
            patch(
                "imas_codex.discovery.base.llm.acall_llm_structured", side_effect=_llm
            ),
            patch("imas_codex.llm.prompt_loader.render_prompt", return_value="prompt"),
            patch(
                "imas_codex.standard_names.workers._hybrid_search_neighbours",
                return_value=[],
            ),
            patch("imas_codex.settings.get_model", return_value="refine-seat"),
            patch(
                "imas_codex.standard_names.graph_ops.stop_refine_name_attempt",
                return_value="reviewed",
            ),
            patch(_GC_WORKERS_PATH, return_value=_mock_gc()),
        ):
            await process_refine_name_batch(
                [item], _mock_budget_manager(), asyncio.Event()
            )

        if escalates:
            assert seen["model"] == DEFAULT_ESCALATION_MODEL
        else:
            assert seen["model"] == "refine-seat"

    @pytest.mark.asyncio
    async def test_attempt_number_and_prior_reason_are_logged(self, caplog):
        from imas_codex.standard_names.workers import process_refine_name_batch

        item = _refine_item(refine_attempts=2, refine_stop_reason="transient_failure")

        with (
            patch(
                "imas_codex.discovery.base.llm.acall_llm_structured",
                side_effect=RuntimeError("boom"),
            ),
            patch("imas_codex.llm.prompt_loader.render_prompt", return_value="prompt"),
            patch(
                "imas_codex.standard_names.workers._hybrid_search_neighbours",
                return_value=[],
            ),
            patch("imas_codex.settings.get_model", return_value="refine-seat"),
            patch(
                "imas_codex.standard_names.graph_ops.stop_refine_name_attempt",
                return_value="reviewed",
            ),
            patch(_GC_WORKERS_PATH, return_value=_mock_gc()),
            caplog.at_level("INFO"),
        ):
            await process_refine_name_batch(
                [item], _mock_budget_manager(), asyncio.Event()
            )

        assert "attempt 2/3" in caplog.text
        assert "previous stop: transient_failure" in caplog.text


class TestRecoveryRules:
    """Who gets the rotations back, and who does not."""

    def test_rescore_clears_the_diagnosis_but_not_the_rotations(self):
        from imas_codex.standard_names.graph_ops import stage_name_for_rescore

        gc = _mock_gc([{"prior_stage": "exhausted"}])
        with _patch_gc(gc):
            stage_name_for_rescore("a", run_id="r")

        cypher = " ".join(gc.query.call_args.args[0].split())
        assert "sn.refine_stop_reason = null" in cypher
        # Refunding the budget here would re-open the paid loop for a name
        # that scores low again; a rescore only buys a fresh quorum draw.
        assert "sn.refine_attempts" not in cypher

    def test_a_name_steering_edit_refunds_the_rotations(self):
        from imas_codex.standard_names.edit import _stamp_edit_fields

        gc = _mock_gc([])
        _stamp_edit_fields(
            gc,
            "a",
            edit_mode="hint",
            name_hint="prefer the shorter carrier",
            docs_hint=None,
            edit_reason="r",
            edit_origin="human",
            edit_scope="name",
            edit_status="open",
            run_id="run",
        )

        cypher = " ".join(gc.query.call_args.args[0].split())
        assert (
            "sn.refine_attempts = CASE WHEN $name_hint IS NULL "
            "THEN sn.refine_attempts ELSE 0 END" in cypher
        )
