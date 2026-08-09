"""Tests for the review_docs pipeline stage transitions.

Covers:
- Claim eligibility: only docs_stage='drafted' nodes are claimed
- persist_reviewed_docs three-way stage decision
  (accepted / reviewed / exhausted)
- Token-mismatch no-op
- Reviewer docs fields are written to graph
- Name reviewer fields are unchanged after docs review
- Failed release reverts claim on LLM error
- Worker streams per-item progress
"""

from __future__ import annotations

import asyncio
import json
from contextlib import contextmanager
from typing import Any
from unittest.mock import MagicMock, patch

import pytest

# =============================================================================
# Shared helpers / paths
# =============================================================================

_GC_PATH = "imas_codex.standard_names.graph_ops.GraphClient"


def _mock_gc_query(return_values: list[list[dict]] | None = None):
    """Return a mock GraphClient whose .query() returns successive values."""
    gc = MagicMock()
    gc.__enter__ = MagicMock(return_value=gc)
    gc.__exit__ = MagicMock(return_value=False)
    if return_values is not None:
        gc.query = MagicMock(side_effect=return_values)
    else:
        gc.query = MagicMock(return_value=[])
    return gc


@contextmanager
def _patch_gc(gc):
    with patch(_GC_PATH, return_value=gc):
        yield


def _make_docs_item(
    sn_id: str = "electron_temperature",
    docs_stage: str = "drafted",
    docs_chain_length: int = 0,
    claim_token: str = "tok-review-docs-123",
    **overrides: Any,
) -> dict[str, Any]:
    """Build a claimed-item dict as returned by claim_review_docs_batch."""
    item: dict[str, Any] = {
        "id": sn_id,
        "name": sn_id,
        "description": "Electron temperature profile",
        "documentation": "The electron temperature $T_e$.",
        "kind": "scalar",
        "unit": "eV",
        "cluster_id": None,
        "physics_domain": ["core_profiles"],
        "validation_status": "valid",
        "docs_stage": docs_stage,
        "docs_chain_length": docs_chain_length,
        "claim_token": claim_token,
    }
    item.update(overrides)
    return item


def _mock_budget_manager() -> MagicMock:
    from types import SimpleNamespace

    mgr = MagicMock()
    lease = MagicMock()
    lease.charge_event = MagicMock(return_value=SimpleNamespace(overspend=0.0))
    lease.release_unused = MagicMock(return_value=0.0)
    mgr.reserve = MagicMock(return_value=lease)
    return mgr


# =============================================================================
# 1. Claim eligibility
# =============================================================================


class TestClaimOnlyDraftedDocs:
    """claim_review_docs_batch uses docs_stage='drafted' as gate."""

    def test_claim_only_drafted_docs(self):
        """The claim WHERE clause gates on docs_stage='drafted'."""
        from imas_codex.standard_names.graph_ops import _claim_sn_atomic

        captured: list[str] = []

        def _fake_claim_sn_atomic(
            *,
            eligibility_where: str,
            **kwargs: Any,
        ) -> list[dict]:
            captured.append(eligibility_where)
            return []

        with patch(
            "imas_codex.standard_names.graph_ops._claim_sn_atomic",
            side_effect=_fake_claim_sn_atomic,
        ):
            from imas_codex.standard_names.graph_ops import (
                claim_review_docs_batch,
            )

            claim_review_docs_batch(batch_size=5)

        assert len(captured) == 1
        where = captured[0]
        assert "docs_stage" in where
        assert "'drafted'" in where

    def test_claim_does_not_transition_stage(self):
        """Stage is NOT transitioned at claim time (stage_field is None/not set)."""
        kwargs_captured: list[dict] = []

        def _fake_claim_sn_atomic(**kwargs: Any) -> list[dict]:
            kwargs_captured.append(kwargs)
            return []

        with patch(
            "imas_codex.standard_names.graph_ops._claim_sn_atomic",
            side_effect=_fake_claim_sn_atomic,
        ):
            from imas_codex.standard_names.graph_ops import (
                claim_review_docs_batch,
            )

            claim_review_docs_batch(batch_size=5)

        assert kwargs_captured
        kw = kwargs_captured[0]
        assert kw.get("stage_field") is None or kw.get("to_stage") is None


class TestClaimSkipsPendingDocs:
    """SN with docs_stage='pending' must NOT be claimed."""

    def test_claim_skips_pending_docs(self):
        """docs_stage='pending' is excluded from WHERE clause."""
        captured: list[str] = []

        def _fake_claim_sn_atomic(
            *,
            eligibility_where: str,
            **kwargs: Any,
        ) -> list[dict]:
            captured.append(eligibility_where)
            return []

        with patch(
            "imas_codex.standard_names.graph_ops._claim_sn_atomic",
            side_effect=_fake_claim_sn_atomic,
        ):
            from imas_codex.standard_names.graph_ops import (
                claim_review_docs_batch,
            )

            claim_review_docs_batch(batch_size=5)

        assert captured
        where = captured[0]
        # The gate is strictly on 'drafted' — 'pending' is not included
        assert "'pending'" not in where
        assert "'drafted'" in where


def test_docs_shortfall_blocks_refine_claim() -> None:
    """An unresolved aggregate score cannot spend a docs refinement rotation."""
    captured: list[str] = []

    def _fake_claim_sn_atomic(*, eligibility_where: str, **_kwargs: Any) -> list:
        captured.append(eligibility_where)
        return []

    with patch(
        "imas_codex.standard_names.graph_ops._claim_sn_atomic",
        side_effect=_fake_claim_sn_atomic,
    ):
        from imas_codex.standard_names.graph_ops import claim_refine_docs_batch

        claim_refine_docs_batch()

    assert captured
    assert "sn.docs_review_resolution_method IS NOT NULL" in captured[0]
    assert "sn.docs_review_quorum_shortfall IS NULL" in captured[0]


# =============================================================================
# 1b. Claim-race resolution (the docs-cost root cause)
# =============================================================================


class TestVerifyDocsClaimWinners:
    """_verify_docs_claim_winners drops claim-race losers before LLM spend.

    Root cause of the docs-cost amplification: concurrent pool replicas each
    bind the same eligible node at MATCH time; the lock-serialised SET lets the
    LAST claim_token win, but every replica still fires its (paid) LLM call. The
    verifier re-reads committed state and keeps only nodes that still hold OUR
    token, so a losing replica spends zero LLM calls.
    """

    def test_drops_items_whose_token_lost(self):
        """Only nodes still holding our token survive verification."""
        from imas_codex.standard_names.graph_ops import _verify_docs_claim_winners

        items = [
            {"id": "won_a", "claim_token": "tok-1"},
            {"id": "lost_b", "claim_token": "tok-1"},  # token overwritten by racer
            {"id": "won_c", "claim_token": "tok-1"},
        ]
        # Graph reports only won_a + won_c still carry tok-1 at the eligible stage.
        gc = _mock_gc_query(return_values=[[{"id": "won_a"}, {"id": "won_c"}]])

        with _patch_gc(gc):
            survivors = _verify_docs_claim_winners(items, eligible_stage="drafted")

        assert [it["id"] for it in survivors] == ["won_a", "won_c"]
        # The verification query gates on token + eligible stage.
        cypher = gc.query.call_args_list[0][0][0]
        assert "sn.claim_token = $token" in cypher
        assert "sn.docs_stage = $eligible_stage" in cypher
        kwargs = gc.query.call_args_list[0][1]
        assert kwargs["token"] == "tok-1"
        assert kwargs["eligible_stage"] == "drafted"

    def test_no_query_when_all_survive_is_still_correct(self):
        """When every claimed node wins, all items are returned unchanged."""
        from imas_codex.standard_names.graph_ops import _verify_docs_claim_winners

        items = [
            {"id": "a", "claim_token": "tok-2"},
            {"id": "b", "claim_token": "tok-2"},
        ]
        gc = _mock_gc_query(return_values=[[{"id": "a"}, {"id": "b"}]])

        with _patch_gc(gc):
            survivors = _verify_docs_claim_winners(items, eligible_stage="refining")

        assert [it["id"] for it in survivors] == ["a", "b"]

    def test_empty_items_short_circuit(self):
        """Empty claim list returns immediately without a graph round-trip."""
        from imas_codex.standard_names.graph_ops import _verify_docs_claim_winners

        gc = _mock_gc_query(return_values=[])
        with _patch_gc(gc):
            assert _verify_docs_claim_winners([], eligible_stage="drafted") == []
        gc.query.assert_not_called()


class TestPersistReviewedDocsConcurrentWinNoOp:
    """persist_reviewed_docs is a no-op when a racer already left 'drafted'."""

    def test_concurrent_transition_is_noop(self):
        """SET RETURNs no row (node no longer 'drafted') → returns ''."""
        gc = _mock_gc_query(
            return_values=[
                [{"docs_chain_length": 0}],  # readback still sees 'drafted'
                [],  # SET matched nothing — concurrent reviewer won the race
            ]
        )

        with _patch_gc(gc):
            from imas_codex.standard_names.graph_ops import persist_reviewed_docs

            result = persist_reviewed_docs(
                sn_id="electron_temperature",
                claim_token="tok",
                score=0.95,
                model="m",
                min_score=0.75,
                rotation_cap=3,
                resolution_method="quorum_consensus",
                reviewer_chain_size=3,
            )

        assert result == ""
        # The SET must re-assert docs_stage='drafted' as the CAS guard.
        set_cypher = gc.query.call_args_list[1][0][0]
        assert "sn.docs_stage = 'drafted'" in set_cypher


# =============================================================================
# 2. persist_reviewed_docs — three-way stage decision
# =============================================================================


class TestPersistToAccepted:
    def test_persist_to_accepted(self):
        """verdict='accept' + score >= min_score → docs_stage='accepted'."""
        gc = _mock_gc_query(
            return_values=[
                [{"docs_chain_length": 0}],  # readback
                [{"id": "electron_temperature"}],  # SET write committed
            ]
        )

        with _patch_gc(gc):
            from imas_codex.standard_names.graph_ops import persist_reviewed_docs

            result = persist_reviewed_docs(
                sn_id="electron_temperature",
                claim_token="tok-123",
                score=0.9,
                scores={"description_quality": 18},
                comments="Excellent docs.",
                comments_per_dim={"description_quality": "Clear"},
                model="test/model",
                min_score=0.75,
                rotation_cap=3,
                resolution_method="quorum_consensus",
                reviewer_chain_size=3,
            )

        assert result == "accepted"


class TestPersistToReviewedLowScore:
    def test_persist_to_reviewed_low_score(self):
        """score=0.5, docs_chain_length=0, rotation_cap=3 → 'reviewed'."""
        gc = _mock_gc_query(
            return_values=[
                [{"docs_chain_length": 0}],
                [{"id": "test_name"}],
            ]
        )

        with _patch_gc(gc):
            from imas_codex.standard_names.graph_ops import persist_reviewed_docs

            result = persist_reviewed_docs(
                sn_id="test_name",
                claim_token="tok",
                score=0.5,
                model="m",
                min_score=0.75,
                rotation_cap=3,
                resolution_method="quorum_consensus",
                reviewer_chain_size=3,
            )

        assert result == "reviewed"


class TestPersistToExhaustedAtCap:
    def test_persist_to_reviewed_at_escalator_attempt(self):
        """score=0.5, docs_chain_length=2, rotation_cap=3 → 'reviewed'.

        Returning 'exhausted' here pre-empts the escalated attempt in
        process_refine_docs_batch, which fires at
        docs_chain_length == rotation_cap-1.  The SN must stay 'reviewed'
        so that final escalated refine can run.
        """
        gc = _mock_gc_query(
            return_values=[
                [{"docs_chain_length": 2}],
                [{"id": "test_name"}],
            ]
        )

        with _patch_gc(gc):
            from imas_codex.standard_names.graph_ops import persist_reviewed_docs

            result = persist_reviewed_docs(
                sn_id="test_name",
                claim_token="tok",
                score=0.5,
                model="m",
                min_score=0.75,
                rotation_cap=3,
                resolution_method="quorum_consensus",
                reviewer_chain_size=3,
            )

        assert result == "reviewed"

    def test_persist_to_exhausted_post_escalator(self):
        """score=0.5, docs_chain_length=3, rotation_cap=3 → 'exhausted'.

        After the escalated final refine has produced a chain=3 SN,
        the next review step must mark it exhausted.
        """
        gc = _mock_gc_query(
            return_values=[
                [{"docs_chain_length": 3}],
                [{"id": "test_name"}],
            ]
        )

        with _patch_gc(gc):
            from imas_codex.standard_names.graph_ops import persist_reviewed_docs

            result = persist_reviewed_docs(
                sn_id="test_name",
                claim_token="tok",
                score=0.5,
                model="m",
                min_score=0.75,
                rotation_cap=3,
                resolution_method="quorum_consensus",
                reviewer_chain_size=3,
            )

        assert result == "exhausted"


class TestPersistToReviewedBelowCap:
    def test_persist_to_reviewed_below_cap(self):
        """score=0.5, docs_chain_length=1, rotation_cap=3 → 'reviewed' (not yet at cap)."""
        gc = _mock_gc_query(
            return_values=[
                [{"docs_chain_length": 1}],
                [{"id": "test_name"}],
            ]
        )

        with _patch_gc(gc):
            from imas_codex.standard_names.graph_ops import persist_reviewed_docs

            result = persist_reviewed_docs(
                sn_id="test_name",
                claim_token="tok",
                score=0.5,
                model="m",
                min_score=0.75,
                rotation_cap=3,
                resolution_method="quorum_consensus",
                reviewer_chain_size=3,
            )

        assert result == "reviewed"


class TestAcceptOverridesChainLengthAtCap:
    def test_accept_overrides_chain_length_at_cap(self):
        """docs_chain_length=3, rotation_cap=3 → 'accepted' (acceptance wins)."""
        gc = _mock_gc_query(
            return_values=[
                [{"docs_chain_length": 3}],
                [{"id": "test_name"}],
            ]
        )

        with _patch_gc(gc):
            from imas_codex.standard_names.graph_ops import persist_reviewed_docs

            result = persist_reviewed_docs(
                sn_id="test_name",
                claim_token="tok",
                score=0.9,
                model="m",
                min_score=0.75,
                rotation_cap=3,
                resolution_method="quorum_consensus",
                reviewer_chain_size=3,
            )

        assert result == "accepted"


@pytest.mark.parametrize(
    "resolution_method,reviewer_chain_size,expected_stage,has_shortfall",
    [
        ("quorum_consensus", 3, "accepted", False),
        ("authoritative_escalation", 3, "accepted", False),
        ("max_cycles_reached", 3, "reviewed", True),
        ("single_review", 3, "reviewed", True),
        (None, None, "reviewed", True),
        ("max_cycles_reached", 2, "accepted", False),
    ],
)
def test_docs_acceptance_requires_canonical_quorum_authority(
    resolution_method: str | None,
    reviewer_chain_size: int | None,
    expected_stage: str,
    has_shortfall: bool,
) -> None:
    """Docs acceptance follows the shared name-axis quorum policy exactly."""
    gc = _mock_gc_query(
        return_values=[
            [{"docs_chain_length": 0}],
            [{"id": "test_name"}],
        ]
    )

    with _patch_gc(gc):
        from imas_codex.standard_names.graph_ops import persist_reviewed_docs

        result = persist_reviewed_docs(
            sn_id="test_name",
            claim_token="tok",
            score=0.95,
            model="m",
            min_score=0.85,
            resolution_method=resolution_method,
            reviewer_chain_size=reviewer_chain_size,
            skip_review_node=True,
        )

    assert result == expected_stage
    write_kwargs = gc.query.call_args_list[1].kwargs
    assert bool(write_kwargs["quorum_shortfall"]) is has_shortfall
    assert write_kwargs["resolution_method"] == resolution_method
    cypher = gc.query.call_args_list[1].args[0]
    assert "docs_review_resolution_method" in cypher
    assert "docs_review_quorum_shortfall_at" in cypher


def test_fresh_docs_quorum_clears_a_prior_shortfall() -> None:
    """A quorate result writes null marker fields instead of retaining residue."""
    gc = _mock_gc_query(
        return_values=[
            [{"docs_chain_length": 0}],
            [{"id": "test_name"}],
        ]
    )
    with _patch_gc(gc):
        from imas_codex.standard_names.graph_ops import persist_reviewed_docs

        result = persist_reviewed_docs(
            sn_id="test_name",
            claim_token="tok",
            score=0.95,
            model="m",
            resolution_method="authoritative_escalation",
            reviewer_chain_size=3,
            skip_review_node=True,
        )

    assert result == "accepted"
    write_kwargs = gc.query.call_args_list[1].kwargs
    assert write_kwargs["quorum_shortfall"] is None


def test_missing_authority_preserves_exhaustion_but_blocks_refinement() -> None:
    """A low score keeps its lifecycle decision while recording the shortfall."""
    gc = _mock_gc_query(
        return_values=[
            [{"docs_chain_length": 3}],
            [{"id": "test_name"}],
        ]
    )
    with _patch_gc(gc):
        from imas_codex.standard_names.graph_ops import persist_reviewed_docs

        result = persist_reviewed_docs(
            sn_id="test_name",
            claim_token="tok",
            score=0.5,
            model="m",
            rotation_cap=3,
            skip_review_node=True,
        )

    assert result == "exhausted"
    write_kwargs = gc.query.call_args_list[1].kwargs
    assert write_kwargs["quorum_shortfall"] == "review carried no resolution method"


class TestPersistTokenMismatchNoOp:
    def test_persist_token_mismatch_no_op(self):
        """Wrong claim_token → returns '' and no SET is executed."""
        # Token mismatch: first query returns empty rows
        gc = _mock_gc_query(return_values=[[]])

        with _patch_gc(gc):
            from imas_codex.standard_names.graph_ops import persist_reviewed_docs

            result = persist_reviewed_docs(
                sn_id="test_name",
                claim_token="wrong-token",
                score=0.9,
                model="m",
                min_score=0.75,
                rotation_cap=3,
                resolution_method="quorum_consensus",
                reviewer_chain_size=3,
            )

        assert result == ""
        # Only one query (the readback) should have been called
        assert gc.query.call_count == 1


class TestPersistWritesReviewerDocsFields:
    def test_persist_writes_reviewer_docs_fields(self):
        """All reviewer_*_docs fields are populated in the SET call."""
        gc = _mock_gc_query(
            return_values=[
                [{"docs_chain_length": 0}],
                [{"id": "electron_temperature"}],
            ]
        )

        with _patch_gc(gc):
            from imas_codex.standard_names.graph_ops import persist_reviewed_docs

            persist_reviewed_docs(
                sn_id="electron_temperature",
                claim_token="tok",
                score=0.8,
                scores={"description_quality": 16, "documentation_quality": 18},
                comments="Good docs.",
                comments_per_dim={
                    "description_quality": "OK",
                    "documentation_quality": "Great",
                },
                model="openrouter/test/model",
                min_score=0.75,
                rotation_cap=3,
                resolution_method="quorum_consensus",
                reviewer_chain_size=3,
            )

        # Check the SET query kwargs
        set_call = gc.query.call_args_list[1]
        call_kwargs = set_call[1]  # keyword args dict

        assert call_kwargs.get("score") == 0.8
        assert call_kwargs.get("model") == "openrouter/test/model"
        assert call_kwargs.get("comments") == "Good docs."

        # Scores and comments_per_dim should be JSON strings
        scores_json = call_kwargs.get("scores_json")
        assert scores_json is not None
        scores_parsed = json.loads(scores_json)
        assert scores_parsed.get("description_quality") == 16

        cpd_json = call_kwargs.get("comments_per_dim_json")
        assert cpd_json is not None
        cpd_parsed = json.loads(cpd_json)
        assert cpd_parsed.get("description_quality") == "OK"

        # Cypher should include reviewed_docs_at, docs_stage, reviewer_score_docs
        cypher = set_call[0][0]  # first positional arg is the cypher string
        assert "reviewed_docs_at" in cypher
        assert "docs_stage" in cypher
        assert "reviewer_score_docs" in cypher


class TestPersistDoesNotChangeNameFields:
    def test_persist_does_not_change_name_fields(self):
        """Reviewer name fields are NOT written during docs review."""
        gc = _mock_gc_query(
            return_values=[
                [{"docs_chain_length": 0}],
                [{"id": "electron_temperature"}],
                [],
            ]
        )

        with _patch_gc(gc):
            from imas_codex.standard_names.graph_ops import persist_reviewed_docs

            persist_reviewed_docs(
                sn_id="electron_temperature",
                claim_token="tok",
                score=0.8,
                model="m",
                min_score=0.75,
                rotation_cap=3,
                resolution_method="quorum_consensus",
                reviewer_chain_size=3,
            )

        set_call = gc.query.call_args_list[1]
        cypher = set_call[0][0]

        # Docs review must NOT set name-axis fields.
        # name_stage may appear in the WHERE clause as a filter — only check the SET block.
        set_part = cypher.split("SET", 1)[1] if "SET" in cypher else cypher
        assert "name_stage" not in set_part
        assert "reviewer_score_name" not in cypher
        assert "reviewer_comments_name" not in cypher
        assert "reviewed_name_at" not in cypher


class TestExactDocsRescoreStaging:
    def test_stages_only_aggregate_docs_decision_fields(self) -> None:
        """CAS staging preserves content, depth, and review-node history."""
        gc = _mock_gc_query(
            return_values=[
                [
                    {
                        "prior_stage": "accepted",
                        "description": "Preserved description",
                        "documentation": "Preserved documentation",
                    }
                ]
            ]
        )
        with _patch_gc(gc):
            from imas_codex.standard_names.graph_ops import stage_docs_for_rescore

            result = stage_docs_for_rescore("test_name", run_id="exact-docs-run")

        assert result == {
            "ok": True,
            "sn_id": "test_name",
            "prior_stage": "accepted",
            "run_id": "exact-docs-run",
            "dry_run": False,
        }
        cypher = gc.query.call_args.args[0]
        assert "sn.docs_stage IN ['accepted', 'reviewed', 'exhausted']" in cypher
        assert "sn.claim_token IS NULL" in cypher
        assert "sn.drain_scope_id IS NULL" in cypher
        assert "sn.drain_scope_claimed_at IS NULL" in cypher
        assert "sn.drain_claim_scope_id IS NULL" in cypher
        assert "sn.run_id = $run_id" in cypher
        set_clause = cypher.split("SET", 1)[1].split("RETURN", 1)[0]
        assert "sn.description" not in set_clause
        assert "sn.documentation" not in set_clause
        assert "sn.docs_chain_length" not in set_clause
        assert "StandardNameReview" not in cypher

    def test_claimed_record_fails_closed_without_second_write(self) -> None:
        gc = _mock_gc_query(
            return_values=[
                [],
                [
                    {
                        "name_stage": "accepted",
                        "docs_stage": "reviewed",
                        "claim_token": "owned",
                        "claimed_at": "now",
                    }
                ],
            ]
        )
        with _patch_gc(gc):
            from imas_codex.standard_names.graph_ops import stage_docs_for_rescore

            result = stage_docs_for_rescore("test_name", run_id="exact-docs-run")

        assert result["ok"] is False
        assert "claimed=True" in result["reason"]
        assert gc.query.call_count == 2

    def test_empty_scope_is_refused_without_graph_access(self) -> None:
        gc = _mock_gc_query()
        with _patch_gc(gc):
            from imas_codex.standard_names.graph_ops import stage_docs_for_rescore

            result = stage_docs_for_rescore("test_name", run_id="  ")

        assert result["ok"] is False
        gc.query.assert_not_called()


# =============================================================================
# 3. Worker tests
# =============================================================================


class TestFailedReleaseKeepsDrafted:
    def test_failed_release_compares_and_updates_docs_axis(self) -> None:
        """A docs failure must never predicate or mutate the name lifecycle."""
        gc = _mock_gc_query(return_values=[[{"released": 1}]])
        with _patch_gc(gc):
            from imas_codex.standard_names.graph_ops import (
                release_review_docs_failed_claims,
            )

            released = release_review_docs_failed_claims(
                sn_ids=["test_name"],
                claim_token="docs-token",
                from_stage="drafted",
                to_stage="drafted",
            )

        assert released == 1
        cypher = gc.query.call_args.args[0]
        assert "n.docs_stage = $from_stage" in cypher
        assert "n.docs_stage = $to_stage" in cypher
        set_clause = cypher.split("SET", 1)[1]
        assert "name_stage" not in set_clause

    def test_failed_release_keeps_drafted(self, mock_llm):
        """LLM error path calls release_review_docs_failed_claims with from/to_stage='drafted'."""
        _ = mock_llm  # consume fixture — will raise RuntimeError for missing response

        release_calls: list[dict] = []

        def _fake_release(**kwargs):
            release_calls.append(kwargs)
            return 1

        with (
            patch(
                "imas_codex.settings.get_sn_review_docs_models",
                return_value=["openrouter/test/model"],
            ),
            patch(
                "imas_codex.llm.prompt_loader.render_prompt",
                return_value="Review.",
            ),
            patch(
                "imas_codex.standard_names.graph_ops.release_review_docs_failed_claims",
                side_effect=_fake_release,
            ),
        ):
            from imas_codex.standard_names.workers import process_review_docs_batch

            items = [_make_docs_item(claim_token="tok-failed")]
            mgr = _mock_budget_manager()

            result = asyncio.run(process_review_docs_batch(items, mgr, asyncio.Event()))

        assert result == 0  # nothing processed on error
        assert len(release_calls) == 1
        rc = release_calls[0]
        assert rc.get("from_stage") == "drafted"
        assert rc.get("to_stage") == "drafted"
        assert "tok-failed" in str(rc.get("claim_token", ""))


class TestWorkerStreamsPerItemDocs:
    def test_worker_streams_per_item(self, mock_llm):
        """process_review_docs_batch returns one processed item per SN."""
        from imas_codex.standard_names.models import (
            StandardNameQualityCommentsDocs,
            StandardNameQualityReviewDocs,
            StandardNameQualityScoreDocs,
        )

        for i in range(3):
            mock_llm.add_response(
                "review_docs",
                response=StandardNameQualityReviewDocs(
                    source_id=f"sn_{i}",
                    standard_name=f"sn_{i}",
                    scores=StandardNameQualityScoreDocs(
                        description_quality=16,
                        documentation_quality=18,
                        completeness=17,
                        physics_accuracy=16,
                    ),
                    reasoning=f"Good docs {i}.",
                ),
            )

        items = [
            _make_docs_item(sn_id=f"sn_{i}", claim_token=f"tok-{i}") for i in range(3)
        ]

        with (
            patch(
                "imas_codex.settings.get_sn_review_docs_models",
                return_value=["openrouter/test/model"],
            ),
            patch(
                "imas_codex.standard_names.graph_ops.write_reviews",
            ),
            patch(
                "imas_codex.standard_names.graph_ops.persist_reviewed_docs",
                return_value="accepted",
            ),
            patch(
                "imas_codex.llm.prompt_loader.render_prompt",
                return_value="documentation quality review.",
            ),
        ):
            from imas_codex.standard_names.workers import process_review_docs_batch

            result = asyncio.run(
                process_review_docs_batch(
                    items, _mock_budget_manager(), asyncio.Event()
                )
            )

        assert result == 3
