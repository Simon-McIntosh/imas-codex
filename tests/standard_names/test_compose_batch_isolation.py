"""A compose batch survives one unusable candidate and one dropped connection.

Production evidence: a full-DD run lost 35 batches. Sixteen carried a grammar
refusal from ``compose_name`` for a single candidate, and fifteen carried a
transport fault from the compose call. Each failure discarded every candidate
in the paid response, because the exception left ``compose_batch`` and the
pool recorded the batch as failed.

Both defects are per-source or per-call outcomes, not batch outcomes. These
tests drive the batch processor over fakes and assert that the valid sibling
still persists, and that a dropped connection is retried under the LLM retry
policy rather than raised.
"""

from __future__ import annotations

import asyncio
from types import SimpleNamespace
from typing import Any
from unittest.mock import AsyncMock, MagicMock, patch

import httpx
import pytest

_GOOD_PATH = "equilibrium/time_slice/profiles_1d/psi"
_REFUSED_PATH = "equilibrium/time_slice/profiles_1d/psi_ratio"
_GOOD_NAME = "electron_temperature"
# The first parse in this module imports the ISN grammar and token tables,
# which costs several seconds before any assertion runs.
pytestmark = pytest.mark.timeout(180)

_GRAMMAR_MESSAGE = (
    "ISN rejected operator chain ratio: name "
    "'ratio_of_beam_launched_power_to_launched_power' round-trips textually "
    "but changes the semantic IR"
)


def _make_batch_item(path: str) -> dict[str, Any]:
    return {
        "id": f"dd:{path}",
        "path": path,
        "claim_token": "winner",
        "claim_seq": 3,
        "description": "Poloidal flux",
        "physics_domain": "equilibrium",
        "unit": "Wb",
        "data_type": "FLT_1D",
        "cocos_version": 11,
        "dd_version": "4.0.0",
    }


class _FakeSegments:
    def __init__(self, base_token: str):
        self.base_token = base_token
        self.base_kind = "quantity"
        self.locus_token = ""
        self.locus_relation = ""
        self.locus_type = ""

    def model_dump(self) -> dict[str, Any]:
        return {"base_token": self.base_token}


class _FakeCandidate:
    def __init__(self, name: str, source_id: str, *, raises: bool = False):
        self.source_id = source_id
        self.description = "Poloidal flux at the plasma boundary"
        self.kind = "scalar"
        self.dd_paths = [source_id]
        self.reason = ""
        self.segments = _FakeSegments("ratio_of" if raises else name)
        self._name = name
        self._raises = raises

    def compose_name(self) -> str:
        if self._raises:
            raise ValueError(_GRAMMAR_MESSAGE)
        return self._name


class _FakeBatchResult:
    def __init__(self, candidates: list[_FakeCandidate]):
        self.candidates = candidates
        self.vocab_gaps: list[Any] = []
        self.attachments: list[Any] = []
        self.skipped: list[Any] = []
        self.dd_gaps: list[Any] = []


class _FakeLLMOut:
    def __init__(self, result: _FakeBatchResult, cost: float = 0.01, tokens: int = 100):
        self._result = result
        self._cost = cost
        self._tokens = tokens
        self.input_tokens = tokens
        self.output_tokens = tokens // 2
        self.cache_read_tokens = 0
        self.cache_creation_tokens = 0

    def __iter__(self):
        return iter((self._result, self._cost, self._tokens))


class _FakeLease:
    def charge_event(self, cost, event):
        return MagicMock(overspend=0)

    def release_unused(self):
        pass


class _FakeBudgetManager:
    def __init__(self) -> None:
        self.run_id = "test-run-id"

    def reserve(self, amount, phase=""):
        return _FakeLease()


@pytest.fixture
def compose_deps():
    """Patch every collaborator compose_batch reaches outside the batch logic."""
    mock_gc = MagicMock()
    mock_gc.__enter__ = MagicMock(return_value=mock_gc)
    mock_gc.__exit__ = MagicMock(return_value=False)
    mock_gc.query = MagicMock(return_value=[])

    refusals: list[dict[str, Any]] = []
    persisted: list[dict[str, Any]] = []

    def _record_refusal(source_id, **kwargs):
        refusals.append({"source_id": source_id, **kwargs})
        return True

    def _fake_persist(candidates, **_kwargs):
        persisted.extend(candidates)
        return [f"dd:{candidate['source_id']}" for candidate in candidates]

    patches = [
        patch(
            "imas_codex.standard_names.context.build_compose_context",
            return_value={},
        ),
        patch("imas_codex.settings.get_model", return_value="local/test-model"),
        patch(
            "imas_codex.llm.prompt_loader.render_prompt",
            side_effect=lambda name, *a, **k: "compose prompt",
        ),
        patch(
            "imas_codex.standard_names.worker_tasks.worker_extraction_and_enrichment._enrich_batch_items",
            side_effect=lambda items: None,
        ),
        patch(
            "imas_codex.standard_names.worker_tasks.worker_extraction_and_enrichment._search_nearby_names",
            return_value=[],
        ),
        patch(
            "imas_codex.standard_names.worker_tasks.worker_extraction_and_enrichment._enrich_ids_context",
            return_value=None,
        ),
        patch(
            "imas_codex.standard_names.context.build_domain_vocabulary_preseed",
            return_value="",
        ),
        patch(
            "imas_codex.standard_names.review.themes.extract_reviewer_themes",
            return_value=[],
        ),
        patch(
            "imas_codex.standard_names.example_loader.load_compose_examples",
            return_value=[],
        ),
        patch("imas_codex.graph.client.GraphClient", return_value=mock_gc),
        patch(
            "imas_codex.standard_names.graph_store.graph_claim_release._verify_source_claim_winners",
            side_effect=lambda items, **_kwargs: items,
        ),
        patch(
            "imas_codex.standard_names.worker_tasks.worker_common._persist_description_checked_candidates",
            side_effect=_fake_persist,
        ),
        patch(
            "imas_codex.standard_names.worker_tasks.worker_common._mark_source_prevalidation_failed",
            side_effect=_record_refusal,
        ),
        patch(
            "imas_codex.standard_names.worker_tasks.worker_extraction_and_enrichment._write_composed_name_domains",
            new=AsyncMock(return_value=None),
        ),
        patch(
            "imas_codex.standard_names.worker_tasks.worker_extraction_and_enrichment._retry_attempts",
            return_value=1,
        ),
        patch(
            "imas_codex.standard_names.audits.run_audits",
            return_value=[],
        ),
        patch(
            "imas_codex.standard_names.worker_tasks.worker_common._auto_detect_physical_base_gaps",
            return_value=[],
        ),
    ]
    for p in patches:
        p.start()
    try:
        yield SimpleNamespace(refusals=refusals, persisted=persisted)
    finally:
        patch.stopall()


class TestGrammarRefusalIsolation:
    """One candidate's grammar refusal must not fail the batch."""

    @pytest.mark.asyncio
    async def test_refused_candidate_is_recorded_and_siblings_still_persist(
        self, compose_deps
    ) -> None:
        from imas_codex.standard_names.worker_tasks.worker_compose_batches import (
            compose_batch,
        )

        result = _FakeBatchResult(
            [
                _FakeCandidate("", _REFUSED_PATH, raises=True),
                _FakeCandidate(_GOOD_NAME, _GOOD_PATH),
            ]
        )
        llm_out = _FakeLLMOut(result)

        async def mock_llm(*_args, **_kwargs):
            return llm_out

        with patch("imas_codex.discovery.base.llm.acall_llm_structured", mock_llm):
            count = await compose_batch(
                [_make_batch_item(_REFUSED_PATH), _make_batch_item(_GOOD_PATH)],
                _FakeBudgetManager(),
                asyncio.Event(),
            )

        assert count == 1
        assert [candidate["id"] for candidate in compose_deps.persisted] == [_GOOD_NAME]
        assert len(compose_deps.refusals) == 1
        refusal = compose_deps.refusals[0]
        assert refusal["source_id"] == _REFUSED_PATH
        assert _GRAMMAR_MESSAGE in refusal["reason"]
        assert refusal["claim_token"] == "winner"
        assert refusal["claim_seq"] == 3


class TestTransportFaultIsolation:
    """A dropped connection retries under the LLM policy; it does not fail."""

    @pytest.mark.asyncio
    @pytest.mark.parametrize(
        "transport_error",
        [
            httpx.ConnectError("Cannot connect to host openrouter.ai:443"),
            httpx.ConnectTimeout("timed out"),
        ],
    )
    async def test_transient_connect_error_is_retried(
        self, compose_deps, transport_error
    ) -> None:
        from imas_codex.standard_names.worker_tasks.worker_compose_batches import (
            compose_batch,
        )

        llm_out = _FakeLLMOut(
            _FakeBatchResult([_FakeCandidate(_GOOD_NAME, _GOOD_PATH)])
        )
        calls = 0

        async def mock_llm(*_args, **_kwargs):
            nonlocal calls
            calls += 1
            if calls == 1:
                raise transport_error
            return llm_out

        sleep = AsyncMock(return_value=None)
        with (
            patch("imas_codex.discovery.base.llm.acall_llm_structured", mock_llm),
            patch("asyncio.sleep", sleep),
        ):
            count = await compose_batch(
                [_make_batch_item(_GOOD_PATH)],
                _FakeBudgetManager(),
                asyncio.Event(),
            )

        assert calls == 2
        assert sleep.await_count == 1
        assert count == 1
        assert [candidate["id"] for candidate in compose_deps.persisted] == [_GOOD_NAME]
