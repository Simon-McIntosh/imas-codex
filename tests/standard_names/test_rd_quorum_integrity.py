"""Quorum-integrity guard for the RD-quorum reviewer chain.

A profile that configures ≥2 reviewer models must NOT accept a StandardName on
a single surviving review when a secondary reviewer fails (throttled or empty
response). ``_run_rd_quorum_cycles`` defers such an item — returns ``None`` so
the caller releases the claim back to ``drafted`` — and counts the deferral so
a throttled run is visible in the run summary rather than silently degrading
name acceptance to single-model.

Covers:
- The structured-call double answers the production call surface and honours
  the per-attempt budget hook (a double that drifts from that surface turns
  every guard below green-by-accident or red-by-accident).
- 2-model profile, secondary fails → deferred (None), warned, counted.
- 1-model profile, primary succeeds → single_review, VALID (unchanged).
- 3-model profile, both base cycles succeed, no disagreement → quorum_consensus
  (escalator NOT invoked).
- 3-model profile, only the primary succeeds → deferred (None), counted.
- Derived-parent path (caller passes a 1-model list) → single_review, VALID.
- Caller contract: a deferred (None) review releases the claim to ``drafted``
  and does NOT call persist_reviewed_docs.
"""

from __future__ import annotations

import asyncio
import copy
import inspect
import json
import logging
from types import SimpleNamespace
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from imas_codex.discovery.base.llm import acall_llm_structured, get_model_limits
from imas_codex.standard_names.graph_ops import (
    DOCS_AUTHORITY_PROJECTION_VERSION,
    DocsAuthorityBackfillConflict,
    backfill_docs_review_authority,
    build_docs_review_authority_manifest,
    compute_docs_review_evidence_hash,
)
from imas_codex.standard_names.workers import (
    _run_rd_quorum_cycles,
    admit_docs_review_request,
    prepare_docs_review_request,
    process_review_docs_batch,
    quorum_incomplete_snapshot,
    reset_quorum_incomplete,
)

# The double binds every call against the real structured-LLM signature, so a
# parameter added to the production function needs no edit here while a caller
# keyword that production would reject still raises.
_PRODUCTION_CALL_SIGNATURE = inspect.signature(acall_llm_structured)

_NAMES_DIMS = {"grammar": 16, "semantic": 16, "convention": 16, "completeness": 16}
_DOCS_PARENT_DIMS = {
    "generalization": 16,
    "positioning": 16,
    "physics_accuracy": 16,
    "clarity": 16,
}


def _docs_request_item() -> dict:
    return {
        "id": "electron_temperature",
        "name": "electron_temperature",
        "description": "Electron temperature profile",
        "documentation": "The electron temperature $T_e$.",
        "kind": "scalar",
        "unit": "eV",
        "physics_domain": "core_profiles",
        "source_paths": [],
        "docs_stage": "accepted",
        "run_id": None,
    }


def _request_preparation_patches(*, exposure: float = 1.0):
    return (
        patch(
            "imas_codex.settings.get_sn_review_docs_models",
            return_value=["m0", "m1", "m2"],
        ),
        patch(
            "imas_codex.standard_names.workers._load_docs_review_admission_item",
            side_effect=lambda _sn_id: _docs_request_item(),
        ),
        patch(
            "imas_codex.standard_names.workers._load_docs_review_parent_children",
            return_value=[],
        ),
        patch(
            "imas_codex.standard_names.workers._load_docs_review_examples",
            return_value=[],
        ),
        patch(
            "imas_codex.standard_names.context.fetch_review_neighbours",
            return_value={
                "vector_neighbours": [],
                "same_base_neighbours": [],
                "same_path_neighbours": [],
            },
        ),
        patch(
            "imas_codex.llm.prompt_loader.render_prompt",
            side_effect=lambda template, context: (
                f"{template}:{context['item']['id']}:{context['item']['description']}"
            ),
        ),
        patch(
            "imas_codex.standard_names.workers.model_provider_exposure",
            return_value=exposure,
        ),
        patch(
            "imas_codex.discovery.base.llm.get_catalog_model_info",
            return_value={"max_input_tokens": 32_000},
        ),
    )


def test_docs_request_identity_is_deterministic_and_prices_possible_escalator() -> None:
    item = _docs_request_item()
    patches = _request_preparation_patches(exposure=1.25)
    with (
        patches[0],
        patches[1],
        patches[2],
        patches[3],
        patches[4],
        patches[5],
        patches[6],
        patches[7],
    ):
        first = asyncio.run(prepare_docs_review_request(item))
        second = asyncio.run(prepare_docs_review_request(copy.deepcopy(item)))

    assert first["request_identity"] == second["request_identity"]
    assert first["models"] == ["m0", "m1", "m2"]
    assert first["expected_exposures"] == [1.25, 1.25, 1.25]
    assert first["expected_exposure"] == pytest.approx(3.75)
    assert first["provider_policy_ceiling_is_separate"] is True
    assert first["escalation_input_token_bound"] == 32_000
    actual_prompt = first["escalation_prompt_factory"](
        [{"reasoning": "z" * 100_000, "scores_json": "{}"}]
    )
    serialized_request = json.dumps(
        {
            "messages": [
                {"role": "system", "content": first["system_prompt"]},
                {"role": "user", "content": actual_prompt},
            ],
            "response_schema": first["response_model"].model_json_schema(),
        },
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
    )
    assert len(serialized_request.encode()) + 4_096 <= 32_000


def test_insufficient_docs_exposure_refuses_before_graph_or_provider() -> None:
    item = _docs_request_item()
    stage = MagicMock(side_effect=AssertionError("graph mutation attempted"))
    provider = AsyncMock(side_effect=AssertionError("provider call attempted"))
    patches = _request_preparation_patches(exposure=2.0)
    with (
        patches[0],
        patches[1],
        patches[2],
        patches[3],
        patches[4],
        patches[5],
        patches[6],
        patches[7],
        patch(
            "imas_codex.standard_names.graph_ops.stage_docs_for_rescore",
            new=stage,
        ),
        patch(
            "imas_codex.discovery.base.llm.acall_llm_structured",
            new=provider,
        ),
    ):
        result = asyncio.run(
            admit_docs_review_request(
                item,
                scope_run_id="exact-scope",
                expected_docs_hash="a" * 64,
                expected_review_input_hash="b" * 64,
                tranche_remaining=5.99,
                campaign_remaining=20.0,
            )
        )

    assert result["ok"] is False
    assert result["outcome"] == "insufficient_exposure"
    assert result["expected_exposure"] == pytest.approx(6.0)
    stage.assert_not_called()
    provider.assert_not_awaited()


def test_docs_request_price_drift_changes_identity() -> None:
    item = _docs_request_item()
    stable = _request_preparation_patches(exposure=1.0)
    with (
        stable[0],
        stable[1],
        stable[2],
        stable[3],
        stable[4],
        stable[5],
        stable[6],
        stable[7],
    ):
        before = asyncio.run(prepare_docs_review_request(item))
    drifted = _request_preparation_patches(exposure=1.01)
    with (
        drifted[0],
        drifted[1],
        drifted[2],
        drifted[3],
        drifted[4],
        drifted[5],
        drifted[6],
        drifted[7],
    ):
        after = asyncio.run(prepare_docs_review_request(item))

    assert before["request_identity"] != after["request_identity"]


def test_claimed_docs_request_drift_rolls_back_before_provider_dispatch() -> None:
    item = _docs_request_item()
    scope_run_id = "priced-docs-scope"
    stage = MagicMock(
        return_value={
            "ok": True,
            "sn_id": item["id"],
            "admission_id": "d" * 64,
            "receipt_identity": "e" * 64,
            "review_group_id": "review-group",
        }
    )
    stable = _request_preparation_patches(exposure=1.0)
    with (
        stable[0],
        stable[1],
        stable[2],
        stable[3],
        stable[4],
        stable[5],
        stable[6],
        stable[7],
        patch(
            "imas_codex.standard_names.graph_ops.stage_docs_for_rescore",
            new=stage,
        ),
    ):
        admitted = asyncio.run(
            admit_docs_review_request(
                item,
                scope_run_id=scope_run_id,
                expected_docs_hash="a" * 64,
                expected_review_input_hash="b" * 64,
                tranche_remaining=10.0,
                campaign_remaining=10.0,
            )
        )
    assert admitted["ok"] is True

    claimed = {
        **item,
        "docs_stage": "drafted",
        "run_id": scope_run_id,
        "claim_token": "claim-token",
        "claim_seq": 3,
        "docs_review_admission": "d" * 64,
    }
    rollback = MagicMock(return_value={"ok": True})
    provider_chain = AsyncMock(
        side_effect=AssertionError("provider chain reached after request drift")
    )
    drifted = _request_preparation_patches(exposure=1.01)
    with (
        drifted[0],
        drifted[1],
        drifted[2],
        drifted[3],
        drifted[4],
        drifted[5],
        drifted[6],
        drifted[7],
        patch(
            "imas_codex.standard_names.graph_ops.bind_docs_review_admission_claim",
            return_value={
                "ok": True,
                "admission_id": "d" * 64,
                "review_group_id": "review-group",
            },
        ),
        patch(
            "imas_codex.standard_names.graph_ops.verify_docs_review_admission_request",
            return_value={"ok": False, "outcome": "admission_drift"},
        ),
        patch(
            "imas_codex.standard_names.graph_ops.rollback_docs_rescore_admission",
            new=rollback,
        ),
        patch(
            "imas_codex.standard_names.workers._run_rd_quorum_cycles",
            new=provider_chain,
        ),
    ):
        from imas_codex.standard_names.workers import process_review_docs_batch

        processed = asyncio.run(
            process_review_docs_batch(
                [claimed], _mock_budget_manager(), asyncio.Event()
            )
        )

    assert processed == 0
    rollback.assert_called_once_with("d" * 64, claim_token="claim-token", claim_seq=3)
    provider_chain.assert_not_awaited()


def _prepared_docs_request(*, identity: str = "c" * 64) -> dict:
    return {
        "id": "electron_temperature",
        "models": ["m0", "m1", "m2"],
        "messages": [],
        "response_schema": {},
        "rubric_dims": [
            "description_quality",
            "documentation_quality",
            "completeness",
            "physics_accuracy",
        ],
        "disagreement_threshold": 0.2,
        "reasoning_effort": "low",
        "escalation_reasoning_effort": "medium",
        "expected_exposures": [1.0, 1.0, 2.0],
        "expected_exposure": 4.0,
        "provider_policy_ceiling_is_separate": True,
        "request_identity": identity,
        "response_model": MagicMock(),
        "user_prompt": "review docs",
        "system_prompt": "review system",
        "escalation_prompt_factory": MagicMock(),
    }


def _successful_docs_quorum() -> dict:
    return {
        "records": [
            {
                "id": "electron_temperature:docs:review-group:0",
                "standard_name_id": "electron_temperature",
                "review_group_id": "review-group",
                "review_axis": "docs",
            }
        ],
        "winning_score": 0.95,
        "winning_scores": {"physics_accuracy": 0.95},
        "winning_comments": "sound",
        "winning_comments_per_dim": None,
        "canonical_model": "m0",
        "resolution_method": "quorum_consensus",
        "total_cost": 1.0,
        "total_tokens_in": 10,
        "total_tokens_out": 5,
        "review_group_id": "review-group",
        "dd_gaps": [],
    }


def test_fresh_process_handoff_dispatches_only_after_durable_verification() -> None:
    item = {
        **_docs_request_item(),
        "docs_stage": "drafted",
        "run_id": "priced-docs-scope",
        "claim_token": "claim-token",
        "claim_seq": 7,
        "docs_review_admission": "admission-id",
    }
    prepared = _prepared_docs_request()
    bind = MagicMock(
        return_value={
            "ok": True,
            "admission_id": "admission-id",
            "review_group_id": "review-group",
        }
    )
    verify = MagicMock(return_value={"ok": True, "admission_id": "admission-id"})
    provider_chain = AsyncMock(return_value=_successful_docs_quorum())
    persist = MagicMock(return_value={"ok": True, "stage": "accepted"})

    with (
        patch(
            "imas_codex.standard_names.workers.prepare_docs_review_request",
            new=AsyncMock(return_value=prepared),
        ),
        patch(
            "imas_codex.standard_names.graph_ops.bind_docs_review_admission_claim",
            new=bind,
        ),
        patch(
            "imas_codex.standard_names.graph_ops.verify_docs_review_admission_request",
            new=verify,
        ),
        patch(
            "imas_codex.standard_names.workers._run_rd_quorum_cycles",
            new=provider_chain,
        ),
        patch(
            "imas_codex.standard_names.graph_ops.persist_admitted_docs_review",
            new=persist,
        ),
        patch(
            "imas_codex.standard_names.workers._persist_dd_gap_evidence",
        ),
        patch(
            "imas_codex.standard_names.graph_ops.update_review_aggregates",
        ),
    ):
        processed = asyncio.run(
            process_review_docs_batch([item], _mock_budget_manager(), asyncio.Event())
        )

    assert processed == 1
    bind.assert_called_once()
    verify.assert_called_once()
    provider_chain.assert_awaited_once()
    assert provider_chain.await_args.kwargs["review_group_id"] == "review-group"
    persist.assert_called_once()


def test_missing_durable_admission_refuses_before_provider_dispatch() -> None:
    item = {
        **_docs_request_item(),
        "docs_stage": "drafted",
        "run_id": "priced-docs-scope",
        "claim_token": "claim-token",
        "claim_seq": 1,
        "docs_review_admission": "admission-id",
    }
    provider_chain = AsyncMock(
        side_effect=AssertionError("provider called without durable admission")
    )
    rollback = MagicMock(return_value={"ok": False, "outcome": "missing_admission"})
    release = MagicMock(return_value=1)
    with (
        patch(
            "imas_codex.standard_names.graph_ops.bind_docs_review_admission_claim",
            return_value={"ok": False, "outcome": "missing_admission"},
        ),
        patch(
            "imas_codex.standard_names.graph_ops.rollback_docs_rescore_admission",
            new=rollback,
        ),
        patch(
            "imas_codex.standard_names.graph_ops.release_review_docs_failed_claims",
            new=release,
        ),
        patch(
            "imas_codex.standard_names.workers._run_rd_quorum_cycles",
            new=provider_chain,
        ),
    ):
        processed = asyncio.run(
            process_review_docs_batch([item], _mock_budget_manager(), asyncio.Event())
        )

    assert processed == 0
    provider_chain.assert_not_awaited()
    rollback.assert_called_once()
    release.assert_called_once_with(
        sn_ids=[item["id"]],
        claim_token="claim-token",
        from_stage="drafted",
        to_stage="drafted",
    )


def test_docs_review_cancellation_rolls_back_durable_admission() -> None:
    item = {
        **_docs_request_item(),
        "docs_stage": "drafted",
        "run_id": "priced-docs-scope",
        "claim_token": "claim-token",
        "claim_seq": 1,
        "docs_review_admission": "admission-id",
    }
    stop_event = asyncio.Event()
    stop_event.set()
    rollback = MagicMock(return_value={"ok": True, "outcome": "rolled_back"})
    with patch(
        "imas_codex.standard_names.graph_ops.rollback_docs_rescore_admission",
        new=rollback,
    ):
        processed = asyncio.run(
            process_review_docs_batch([item], _mock_budget_manager(), stop_event)
        )

    assert processed == 0
    rollback.assert_called_once()


def test_admitted_terminal_persistence_noop_rolls_back_partial_reviews() -> None:
    item = {
        **_docs_request_item(),
        "docs_stage": "drafted",
        "run_id": "priced-docs-scope",
        "claim_token": "claim-token",
        "claim_seq": 2,
        "docs_review_admission": "admission-id",
    }
    rollback = MagicMock(return_value={"ok": True, "outcome": "rolled_back"})
    with (
        patch(
            "imas_codex.standard_names.workers.prepare_docs_review_request",
            new=AsyncMock(return_value=_prepared_docs_request()),
        ),
        patch(
            "imas_codex.standard_names.graph_ops.bind_docs_review_admission_claim",
            return_value={
                "ok": True,
                "admission_id": "admission-id",
                "review_group_id": "review-group",
            },
        ),
        patch(
            "imas_codex.standard_names.graph_ops.verify_docs_review_admission_request",
            return_value={"ok": True, "admission_id": "admission-id"},
        ),
        patch(
            "imas_codex.standard_names.workers._run_rd_quorum_cycles",
            new=AsyncMock(return_value=_successful_docs_quorum()),
        ),
        patch(
            "imas_codex.standard_names.graph_ops.persist_admitted_docs_review",
            return_value={"ok": False, "outcome": "terminal_persistence_noop"},
        ),
        patch(
            "imas_codex.standard_names.graph_ops.rollback_docs_rescore_admission",
            new=rollback,
        ),
    ):
        processed = asyncio.run(
            process_review_docs_batch([item], _mock_budget_manager(), asyncio.Event())
        )

    assert processed == 0
    rollback.assert_called_once()


# ---------------------------------------------------------------------------
# Fake structured-LLM plumbing
# ---------------------------------------------------------------------------


class _FakeScores:
    def __init__(self, score: float, dims: dict) -> None:
        self.score = score
        self._dims = dict(dims)

    def model_dump(self) -> dict:
        return dict(self._dims)


class _FakeReviewItem:
    def __init__(self, score: float, dims: dict, reasoning: str = "looks fine") -> None:
        self.scores = _FakeScores(score, dims)
        self.reasoning = reasoning
        self.comments = None


class _FakeBatch:
    def __init__(self, reviews: list) -> None:
        self.reviews = reviews


def _make_acall(responses: list[dict | None]):
    """Build a fake ``acall_llm_structured``.

    ``responses`` is consumed one entry per cycle: a dict ``{"score", "dims"}``
    yields a parseable review; ``None`` yields an empty ``reviews`` batch that
    the chain treats as a FAILED cycle (the throttled / empty-response case).
    Returns ``(acall, calls)`` where ``calls["n"]`` is the invocation count.
    """
    calls = {"n": 0}

    async def _acall(**kwargs):
        bound = _PRODUCTION_CALL_SIGNATURE.bind(**kwargs)
        bound.apply_defaults()
        idx = calls["n"]
        calls["n"] += 1
        # One simulated provider request per call, so the attempt hook fires
        # once with (attempt_index, resolved output allowance) exactly as the
        # production request loop fires it — a hook that raises aborts the
        # cycle before any response is billed, which is what a budget denial
        # on the first attempt does for real.
        before_attempt = bound.arguments.get("before_attempt")
        if before_attempt is not None:
            model = bound.arguments["model"]
            max_tokens = bound.arguments.get("max_tokens") or get_model_limits(
                model
            ).get("max_tokens", 0)
            before_attempt(0, int(max_tokens))
        spec = responses[idx] if idx < len(responses) else None
        if spec is None:
            batch = _FakeBatch([])  # empty → failed cycle
        else:
            batch = _FakeBatch([_FakeReviewItem(spec["score"], spec["dims"])])
        return (batch, 0.01, {})

    return _acall, calls


def _run(
    *,
    models: list[str],
    responses: list[dict | None],
    review_axis: str = "names",
    rubric_dims: tuple[str, ...] = tuple(_NAMES_DIMS),
    run_id: str = "run-x",
):
    acall, calls = _make_acall(responses)
    result = asyncio.run(
        _run_rd_quorum_cycles(
            sn_id="a",
            review_axis=review_axis,
            response_model=None,
            user_prompt="u",
            system_prompt="s",
            models=models,
            disagreement_threshold=0.15,
            rubric_dims=rubric_dims,
            lease=None,
            phase="review",
            acall_llm_structured=acall,
            run_id=run_id,
        )
    )
    return result, calls


# ---------------------------------------------------------------------------
# Double fidelity: the stand-in must answer the real call surface
# ---------------------------------------------------------------------------


class TestStructuredCallDouble:
    def test_accepts_the_production_keyword_surface(self):
        """Every keyword the chain passes is one the real function takes."""
        acall, _calls = _make_acall([{"score": 0.8, "dims": _NAMES_DIMS}])
        chain_kwargs = {
            "model": "m0",
            "messages": [{"role": "user", "content": "u"}],
            "response_model": None,
            "service": "standard-names",
            "reasoning_effort": None,
            "before_attempt": None,
        }
        _PRODUCTION_CALL_SIGNATURE.bind(**chain_kwargs)
        batch, _cost, _tokens = asyncio.run(acall(**chain_kwargs))
        assert len(batch.reviews) == 1

    def test_attempt_hook_runs_and_its_denial_aborts_the_call(self):
        """The hook fires per request; raising stops the call, as in production."""
        seen: list[tuple[int, int]] = []

        def _hook(attempt: int, max_output_tokens: int) -> None:
            seen.append((attempt, max_output_tokens))

        acall, _calls = _make_acall([{"score": 0.8, "dims": _NAMES_DIMS}] * 2)
        asyncio.run(
            acall(
                model="m0",
                messages=[{"role": "user", "content": "u"}],
                response_model=None,
                service="standard-names",
                before_attempt=_hook,
            )
        )
        assert len(seen) == 1
        assert seen[0][0] == 0
        assert seen[0][1] > 0  # priced at the allowance the request carries

        def _deny(attempt: int, max_output_tokens: int) -> None:
            raise RuntimeError("budget denial")

        with pytest.raises(RuntimeError, match="budget denial"):
            asyncio.run(
                acall(
                    model="m0",
                    messages=[{"role": "user", "content": "u"}],
                    response_model=None,
                    service="standard-names",
                    before_attempt=_deny,
                )
            )


# ---------------------------------------------------------------------------
# Direct guard tests
# ---------------------------------------------------------------------------


class TestQuorumCompletenessGuard:
    def test_two_models_secondary_fails_defers(self, caplog):
        """2-model profile with a failed secondary → None, warned, counted."""
        reset_quorum_incomplete("run-x")
        with caplog.at_level(
            logging.WARNING, logger="imas_codex.standard_names.workers"
        ):
            result, calls = _run(
                models=["m0", "m1"],
                responses=[{"score": 0.8, "dims": _NAMES_DIMS}, None],
            )
        assert result is None  # deferred, NOT accepted on a single review
        assert quorum_incomplete_snapshot("run-x") == {"names": 1}
        assert "incomplete" in caplog.text
        reset_quorum_incomplete("run-x")

    def test_single_model_is_valid_single_review(self):
        """1-model profile with a successful cycle → single_review (unchanged)."""
        reset_quorum_incomplete("run-x")
        result, calls = _run(
            models=["m0"],
            responses=[{"score": 0.8, "dims": _NAMES_DIMS}],
        )
        assert result is not None
        assert result["resolution_method"] == "single_review"
        assert quorum_incomplete_snapshot("run-x") == {}
        reset_quorum_incomplete("run-x")

    def test_three_models_two_agree_is_consensus(self):
        """3-model profile, both base cycles agree → quorum_consensus, no escalator."""
        reset_quorum_incomplete("run-x")
        result, calls = _run(
            models=["m0", "m1", "m2"],
            responses=[
                {"score": 0.8, "dims": _NAMES_DIMS},
                {"score": 0.8, "dims": _NAMES_DIMS},
            ],
        )
        assert result is not None
        assert result["resolution_method"] == "quorum_consensus"
        assert calls["n"] == 2  # escalator (cycle 2) NOT invoked
        assert quorum_incomplete_snapshot("run-x") == {}
        reset_quorum_incomplete("run-x")

    def test_three_models_only_primary_succeeds_defers(self):
        """3-model profile, only cycle 0 succeeds → deferred (None), counted."""
        reset_quorum_incomplete("run-x")
        result, calls = _run(
            models=["m0", "m1", "m2"],
            responses=[{"score": 0.8, "dims": _NAMES_DIMS}, None],
        )
        assert result is None
        assert quorum_incomplete_snapshot("run-x") == {"names": 1}
        reset_quorum_incomplete("run-x")

    def test_explicit_single_model_profile_returns_single_review(self):
        """The quorum primitive still describes an explicit one-seat profile."""
        reset_quorum_incomplete("run-x")
        result, calls = _run(
            models=["m0"],
            responses=[{"score": 0.85, "dims": _DOCS_PARENT_DIMS}],
            review_axis="docs",
            rubric_dims=tuple(_DOCS_PARENT_DIMS),
        )
        assert result is not None
        assert result["resolution_method"] == "single_review"
        assert quorum_incomplete_snapshot("run-x") == {}
        reset_quorum_incomplete("run-x")


# ---------------------------------------------------------------------------
# Caller contract: a deferred (None) review releases the claim, no persist
# ---------------------------------------------------------------------------


def _make_docs_item(
    sn_id: str = "electron_temperature", claim_token: str = "tok-defer"
):
    return {
        "id": sn_id,
        "name": sn_id,
        "description": "Electron temperature profile",
        "documentation": "The electron temperature $T_e$.",
        "kind": "scalar",
        "unit": "eV",
        "cluster_id": None,
        "physics_domain": ["core_profiles"],
        "validation_status": "valid",
        "docs_stage": "drafted",
        "docs_chain_length": 0,
        "claim_token": claim_token,
    }


def _mock_budget_manager() -> MagicMock:
    mgr = MagicMock()
    lease = MagicMock()
    lease.charge_event = MagicMock(return_value=SimpleNamespace(overspend=0.0))
    lease.release_unused = MagicMock(return_value=0.0)
    mgr.reserve = MagicMock(return_value=lease)
    mgr.run_id = "run-caller"
    return mgr


class TestDeferredReviewReleasesClaim:
    def test_deferred_quorum_releases_and_skips_persist(self):
        """quorum=None → release claim to drafted, persist_reviewed_docs NOT called."""
        release_calls: list[dict] = []
        persist_calls: list[dict] = []

        def _fake_release(**kwargs):
            release_calls.append(kwargs)
            return 1

        def _fake_persist(**kwargs):
            persist_calls.append(kwargs)
            return "accepted"

        with (
            patch(
                "imas_codex.settings.get_sn_review_docs_models",
                return_value=["m0", "m1"],
            ),
            patch(
                "imas_codex.llm.prompt_loader.render_prompt",
                return_value="Review.",
            ),
            patch(
                "imas_codex.standard_names.workers._run_rd_quorum_cycles",
                new=AsyncMock(return_value=None),
            ),
            patch(
                "imas_codex.standard_names.graph_ops.release_review_docs_failed_claims",
                side_effect=_fake_release,
            ),
            patch(
                "imas_codex.standard_names.graph_ops.persist_reviewed_docs",
                side_effect=_fake_persist,
            ),
        ):
            from imas_codex.standard_names.workers import process_review_docs_batch

            items = [_make_docs_item(claim_token="tok-defer")]
            result = asyncio.run(
                process_review_docs_batch(
                    items, _mock_budget_manager(), asyncio.Event()
                )
            )

        assert result == 0  # nothing advanced
        assert len(persist_calls) == 0  # NOT accepted on a deferred review
        assert len(release_calls) == 1
        rc = release_calls[0]
        assert rc.get("from_stage") == "drafted"
        assert rc.get("to_stage") == "drafted"
        assert "tok-defer" in str(rc.get("claim_token", ""))


# ---------------------------------------------------------------------------
# Exact documentation-authority metadata recovery
# ---------------------------------------------------------------------------


def _authority_state(
    name_id: str = "electron_temperature",
    *,
    current_method: str | None = None,
) -> dict:
    from imas_codex.standard_names.review.audits import compute_review_input_hash

    reviewed_at = "2026-08-09T12:00:00+00:00"
    group_id = "11111111-2222-4333-8444-555555555555"
    properties = {
        "id": name_id,
        "description": "Electron temperature in the plasma core.",
        "documentation": "The electron temperature is measured in eV.",
        "kind": "scalar",
        "unit": "eV",
        "links": [],
        "physical_base": "temperature",
        "subject": "electron",
        "component": None,
        "coordinate": None,
        "position": None,
        "process": None,
        "cocos_transformation_type": None,
        "source_paths": ["dd:core_profiles/profiles_1d/electrons/temperature"],
        "name_stage": "accepted",
        "docs_stage": "accepted",
        "status": "draft",
        "validation_status": "valid",
        "chain_length": 0,
        "docs_chain_length": 0,
        "edit_status": None,
        "claim_token": None,
        "claimed_at": None,
        "run_id": None,
        "drain_scope_id": None,
        "drain_scope_claimed_at": None,
        "drain_claim_scope_id": None,
        "reviewer_score_docs": 0.9,
        "reviewed_docs_at": "2026-08-09T12:00:00.250000+00:00",
        "docs_review_resolution_method": current_method,
        "docs_review_quorum_shortfall": None,
        "docs_review_quorum_shortfall_at": None,
    }
    properties["review_input_hash"] = compute_review_input_hash(properties)
    reviews = [
        {
            "id": f"{name_id}:docs:{group_id}:0",
            "standard_name_id": name_id,
            "review_axis": "docs",
            "review_group_id": group_id,
            "cycle_index": 0,
            "resolution_role": "primary",
            "resolution_method": None,
            "model": "reviewer/primary",
            "model_family": "primary-family",
            "is_canonical": True,
            "score": 0.95,
            "scores_json": '{"clarity": 19}',
            "reviewed_at": reviewed_at,
            "codex_version": "test-build",
            "isn_version": "test-catalog",
        },
        {
            "id": f"{name_id}:docs:{group_id}:1",
            "standard_name_id": name_id,
            "review_axis": "docs",
            "review_group_id": group_id,
            "cycle_index": 1,
            "resolution_role": "secondary",
            "resolution_method": "quorum_consensus",
            "model": "reviewer/secondary",
            "model_family": "secondary-family",
            "is_canonical": False,
            "score": 0.85,
            "scores_json": '{"clarity": 17}',
            "reviewed_at": reviewed_at,
            "codex_version": "test-build",
            "isn_version": "test-catalog",
        },
    ]
    return {"id": name_id, "standard_name": properties, "reviews": reviews}


def _authority_manifest(state: dict) -> dict:
    return build_docs_review_authority_manifest(
        [state["id"]], gc=_AuthorityGraph(_AuthorityTransaction([state]))
    )


class _AuthorityTransaction:
    def __init__(self, states: list[dict]) -> None:
        self.states = {state["id"]: copy.deepcopy(state) for state in states}
        self._original = copy.deepcopy(self.states)
        self.closed = False
        self.committed = False
        self.rolled_back = False
        self.omit_id: str | None = None
        self.extra_id: str | None = None
        self.after_write = None

    def run(self, cypher: str, **params):
        ids = list(params.get("ids") or [])
        if "_docs_authority_lock" in cypher:
            returned = [name_id for name_id in ids if name_id in self.states]
            if self.omit_id in returned:
                returned.remove(self.omit_id)
            if self.extra_id:
                returned.append(self.extra_id)
            return [{"id": name_id} for name_id in sorted(returned)]
        if "properties(sn) AS standard_name" in cypher:
            returned = [
                copy.deepcopy(self.states[name_id])
                for name_id in ids
                if name_id in self.states and name_id != self.omit_id
            ]
            if self.extra_id and self.extra_id in self.states:
                returned.append(copy.deepcopy(self.states[self.extra_id]))
            return sorted(returned, key=lambda state: state["id"])
        if "prior_method" in cypher:
            result = []
            for row in params["rows"]:
                state = self.states.get(row["id"])
                if state is None:
                    continue
                properties = state["standard_name"]
                prior = properties.get("docs_review_resolution_method")
                if prior not in (None, row["resolution_method"]):
                    continue
                properties["docs_review_resolution_method"] = row["resolution_method"]
                result.append({"id": row["id"], "prior_method": prior})
            if self.after_write is not None:
                self.after_write(self.states)
            return result
        raise AssertionError(f"unexpected authority query: {cypher}")

    def commit(self) -> None:
        self.committed = True
        self.closed = True

    def rollback(self) -> None:
        self.states = copy.deepcopy(self._original)
        self.rolled_back = True
        self.closed = True


class _AuthoritySession:
    def __init__(self, transaction: _AuthorityTransaction) -> None:
        self.transaction = transaction

    def __enter__(self):
        return self

    def __exit__(self, *_args):
        return False

    def begin_transaction(self):
        return self.transaction


class _AuthorityGraph:
    def __init__(self, transaction: _AuthorityTransaction) -> None:
        self.transaction = transaction

    def session(self):
        return _AuthoritySession(self.transaction)


class TestDocsAuthorityBackfill:
    def test_builder_emits_deterministic_production_shaped_cohort(self):
        names = [
            "efficiency_of_plant_system",
            "flux_surface_averaged_argon_density_at_plasma_boundary",
            "flux_surface_averaged_carbon_density_at_plasma_boundary",
            "flux_surface_averaged_helium_3_density_at_plasma_boundary",
            "flux_surface_averaged_helium_4_density_at_plasma_boundary",
            "flux_surface_averaged_lithium_density_at_plasma_boundary",
            "flux_surface_averaged_xenon_density_at_plasma_boundary",
            "helium_4_density",
            "ion_state_velocity",
            "ion_state_velocity_due_to_e_cross_b_drift",
            "neutral_state_momentum",
            "oxygen_density",
            "perturbed_plasma_velocity",
            "radial_effective_electron_energy_convection_velocity",
            "radial_minimum_force_of_poloidal_field_coil",
            "ratio_of_hydrogen_density_to_total_hydrogenic_density",
            "toroidal_flux_surface_averaged_helium_3_velocity_at_plasma_boundary",
            "toroidal_plasma_current",
            "total_electron_deposited_power",
            "total_fast_ion_pressure",
            "vertical_coordinate_of_ferritic_element_centroid",
            "xenon_density",
        ]
        states = [_authority_state(name) for name in reversed(names)]
        first_transaction = _AuthorityTransaction(states)
        second_transaction = _AuthorityTransaction(states)

        first = build_docs_review_authority_manifest(
            list(reversed(names)), gc=_AuthorityGraph(first_transaction)
        )
        second = build_docs_review_authority_manifest(
            names, gc=_AuthorityGraph(second_transaction)
        )

        assert first == second
        assert set(first) == {
            "schema",
            "projection_version",
            "manifest_id",
            "rows",
        }
        assert first["projection_version"] == DOCS_AUTHORITY_PROJECTION_VERSION
        assert len(first["manifest_id"]) == 64
        assert [row["id"] for row in first["rows"]] == sorted(names)
        assert len(first["rows"]) == 22
        assert first_transaction.rolled_back is True
        assert second_transaction.rolled_back is True
        for state, row in zip(
            sorted(states, key=lambda item: item["id"]), first["rows"], strict=True
        ):
            assert row["expected_review_evidence_hash"] == (
                compute_docs_review_evidence_hash(state["reviews"])
            )

    def test_exact_quorum_evidence_sets_only_authority_method(self):
        state = _authority_state()
        before = copy.deepcopy(state)
        transaction = _AuthorityTransaction([state])

        result = backfill_docs_review_authority(
            _authority_manifest(state), gc=_AuthorityGraph(transaction)
        )

        assert result["matched"] == 1
        assert result["changed"] == 1
        assert result["postflight_verified"] is True
        assert transaction.committed is True
        after = transaction.states[state["id"]]
        assert after["standard_name"]["docs_review_resolution_method"] == (
            "quorum_consensus"
        )
        before["standard_name"]["docs_review_resolution_method"] = "quorum_consensus"
        assert after == before

    def test_dry_run_executes_postflight_then_rolls_back(self):
        state = _authority_state()
        transaction = _AuthorityTransaction([state])

        result = backfill_docs_review_authority(
            _authority_manifest(state), dry_run=True, gc=_AuthorityGraph(transaction)
        )

        assert result["dry_run"] is True
        assert result["changed"] == 1
        assert transaction.rolled_back is True
        assert (
            transaction.states[state["id"]]["standard_name"][
                "docs_review_resolution_method"
            ]
            is None
        )

    @pytest.mark.parametrize("ids", [[], ["electron_temperature"] * 2])
    def test_empty_or_duplicate_manifest_is_rejected(self, ids):
        transaction = _AuthorityTransaction([_authority_state()])
        with pytest.raises(ValueError):
            build_docs_review_authority_manifest(ids, gc=_AuthorityGraph(transaction))
        assert transaction.committed is False

    def test_projection_version_mismatch_is_rejected_before_transaction(self):
        state = _authority_state()
        manifest = _authority_manifest(state)
        manifest["projection_version"] = DOCS_AUTHORITY_PROJECTION_VERSION + 1
        transaction = _AuthorityTransaction([state])

        with pytest.raises(ValueError, match="projection version"):
            backfill_docs_review_authority(manifest, gc=_AuthorityGraph(transaction))

        assert transaction.committed is False
        assert transaction.rolled_back is False

    @pytest.mark.parametrize("row_fault", ["empty", "duplicate"])
    def test_apply_rejects_empty_or_duplicate_rows_before_transaction(self, row_fault):
        state = _authority_state()
        manifest = _authority_manifest(state)
        manifest["rows"] = [] if row_fault == "empty" else [manifest["rows"][0]] * 2
        transaction = _AuthorityTransaction([state])

        with pytest.raises(ValueError):
            backfill_docs_review_authority(manifest, gc=_AuthorityGraph(transaction))

        assert transaction.committed is False
        assert transaction.rolled_back is False

    @pytest.mark.parametrize("cohort_fault", ["partial", "extra"])
    def test_partial_or_extra_transaction_cohort_is_rejected(self, cohort_fault):
        state = _authority_state()
        transaction = _AuthorityTransaction([state])
        if cohort_fault == "partial":
            transaction.omit_id = state["id"]
        else:
            transaction.extra_id = "unexpected_name"
        with pytest.raises(DocsAuthorityBackfillConflict, match="cohort"):
            backfill_docs_review_authority(
                _authority_manifest(state), gc=_AuthorityGraph(transaction)
            )
        assert transaction.rolled_back is True

    def test_stale_documentation_hash_is_rejected(self):
        state = _authority_state()
        manifest = _authority_manifest(state)
        state["standard_name"]["documentation"] += " Changed."
        transaction = _AuthorityTransaction([state])
        with pytest.raises(DocsAuthorityBackfillConflict, match="content drifted"):
            backfill_docs_review_authority(manifest, gc=_AuthorityGraph(transaction))

    def test_stale_review_input_hash_is_rejected(self):
        state = _authority_state()
        manifest = _authority_manifest(state)
        state["standard_name"]["kind"] = "vector"
        transaction = _AuthorityTransaction([state])
        with pytest.raises(DocsAuthorityBackfillConflict, match="input hash drifted"):
            backfill_docs_review_authority(manifest, gc=_AuthorityGraph(transaction))

    def test_stale_review_fingerprint_is_rejected(self):
        state = _authority_state()
        manifest = _authority_manifest(state)
        state["reviews"][0]["scores_json"] = '{"clarity": 1}'
        transaction = _AuthorityTransaction([state])
        with pytest.raises(DocsAuthorityBackfillConflict, match="evidence drifted"):
            backfill_docs_review_authority(manifest, gc=_AuthorityGraph(transaction))

    @pytest.mark.parametrize(
        "method,cycles",
        [("single_review", [0]), ("max_cycles_reached", [0, 1])],
    )
    def test_nonquorate_resolution_is_rejected_even_when_manifest_claims_quorum(
        self, method, cycles
    ):
        state = _authority_state()
        manifest = _authority_manifest(state)
        state["reviews"] = [
            review for review in state["reviews"] if review["cycle_index"] in cycles
        ]
        state["reviews"][-1]["resolution_method"] = method
        transaction = _AuthorityTransaction([state])
        with pytest.raises(DocsAuthorityBackfillConflict):
            backfill_docs_review_authority(manifest, gc=_AuthorityGraph(transaction))

    def test_aggregate_score_mismatch_is_rejected(self):
        state = _authority_state()
        manifest = _authority_manifest(state)
        state["standard_name"]["reviewer_score_docs"] = 0.99
        transaction = _AuthorityTransaction([state])
        with pytest.raises(DocsAuthorityBackfillConflict, match="aggregate score"):
            backfill_docs_review_authority(manifest, gc=_AuthorityGraph(transaction))

    def test_aggregate_timestamp_drift_is_rejected(self):
        state = _authority_state()
        manifest = _authority_manifest(state)
        state["standard_name"]["reviewed_docs_at"] = "2026-08-09T12:00:07+00:00"
        transaction = _AuthorityTransaction([state])
        with pytest.raises(DocsAuthorityBackfillConflict, match="timestamp"):
            backfill_docs_review_authority(manifest, gc=_AuthorityGraph(transaction))

    def test_lifecycle_drift_is_rejected(self):
        state = _authority_state()
        manifest = _authority_manifest(state)
        state["standard_name"]["name_stage"] = "superseded"
        transaction = _AuthorityTransaction([state])
        with pytest.raises(DocsAuthorityBackfillConflict, match="lifecycle"):
            backfill_docs_review_authority(manifest, gc=_AuthorityGraph(transaction))

    def test_same_manifest_is_idempotent(self):
        state = _authority_state(current_method="quorum_consensus")
        transaction = _AuthorityTransaction([state])
        result = backfill_docs_review_authority(
            _authority_manifest(state), gc=_AuthorityGraph(transaction)
        )
        assert result["changed"] == 0
        assert result["already_applied"] == 1

    def test_authoritative_escalation_is_derived_from_three_canonical_cycles(self):
        state = _authority_state()
        group_id = state["reviews"][0]["review_group_id"]
        state["reviews"][1]["resolution_method"] = None
        state["reviews"].append(
            {
                **state["reviews"][1],
                "id": f"{state['id']}:docs:{group_id}:2",
                "cycle_index": 2,
                "resolution_role": "escalator",
                "resolution_method": "authoritative_escalation",
                "model": "reviewer/escalator",
                "model_family": "escalator-family",
                "score": 0.92,
                "scores_json": '{"clarity": 18.4}',
            }
        )
        state["standard_name"]["reviewer_score_docs"] = 0.92
        manifest = build_docs_review_authority_manifest(
            [state["id"]], gc=_AuthorityGraph(_AuthorityTransaction([state]))
        )
        transaction = _AuthorityTransaction([state])

        result = backfill_docs_review_authority(
            manifest, gc=_AuthorityGraph(transaction)
        )

        assert result["changed"] == 1
        assert (
            transaction.states[state["id"]]["standard_name"][
                "docs_review_resolution_method"
            ]
            == "authoritative_escalation"
        )

    def test_postflight_rejects_collateral_mutation(self):
        state = _authority_state()
        transaction = _AuthorityTransaction([state])
        transaction.after_write = lambda states: states[state["id"]][
            "standard_name"
        ].update(documentation="collateral mutation")

        with pytest.raises(DocsAuthorityBackfillConflict, match="collateral"):
            backfill_docs_review_authority(
                _authority_manifest(state), gc=_AuthorityGraph(transaction)
            )
        assert transaction.rolled_back is True
