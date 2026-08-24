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
from imas_codex.standard_names.workers import (
    _run_rd_quorum_cycles,
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
    review_axis: str = "name",
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
        assert quorum_incomplete_snapshot("run-x") == {"name": 1}
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
        assert quorum_incomplete_snapshot("run-x") == {"name": 1}
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

    def test_lost_primary_does_not_bill_the_secondary(self):
        """A lost primary puts the quorum out of reach — stop before paying more.

        The escalator fires only when both base cycles survived, so with the
        primary gone no further call can carry the item to two reviews. Any
        secondary bought here is billed and then discarded by the guard.
        """
        reset_quorum_incomplete("run-x")
        result, calls = _run(
            models=["m0", "m1", "m2"],
            responses=[None, {"score": 0.8, "dims": _NAMES_DIMS}],
        )
        assert result is None
        assert calls["n"] == 1  # only the primary was attempted
        assert quorum_incomplete_snapshot("run-x") == {"name": 1}
        reset_quorum_incomplete("run-x")


class TestUnfundedSeatIsAttributed:
    """An unaffordable seat calls no provider and raises nothing.

    Left unreported it looks exactly like a provider outage, which sends an
    operator debugging the reviewer chain when the lever is the run's cost
    limit or its reviewer replica count.
    """

    def test_deferral_names_the_budget_and_the_seat(self, caplog):
        starved = MagicMock()
        starved.reserve = MagicMock(return_value=None)

        acall, calls = _make_acall([{"score": 0.8, "dims": _NAMES_DIMS}])
        with (
            patch(
                "imas_codex.standard_names.workers.model_provider_exposure",
                return_value=0.2773,
            ),
            caplog.at_level(
                logging.WARNING, logger="imas_codex.standard_names.workers"
            ),
        ):
            reset_quorum_incomplete("run-x")
            result = asyncio.run(
                _run_rd_quorum_cycles(
                    sn_id="poloidal_electric_field",
                    review_axis="name",
                    response_model=None,
                    user_prompt="u",
                    system_prompt="s",
                    models=["m0", "m1", "m2"],
                    disagreement_threshold=0.15,
                    rubric_dims=tuple(_NAMES_DIMS),
                    lease=None,
                    phase="review_name",
                    acall_llm_structured=acall,
                    budget_manager=starved,
                    run_id="run-x",
                )
            )

        assert result is None
        assert calls["n"] == 0  # no provider was ever reached
        assert "budget refused" in caplog.text
        assert "m0" in caplog.text
        assert "0.2773" in caplog.text
        assert quorum_incomplete_snapshot("run-x") == {"name": 1}
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
