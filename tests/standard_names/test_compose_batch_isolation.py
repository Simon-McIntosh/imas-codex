"""A compose batch survives one unusable candidate and nothing else.

Production evidence: a full-DD run lost 35 batches. Sixteen carried a grammar
refusal from ``compose_name`` for a single candidate, and fifteen carried a
transport fault from the compose call. Each failure discarded every candidate
in the paid response, because the exception left ``compose_batch`` and the
pool recorded the batch as failed.

The grammar refusal is a per-source outcome: the model proposed a name the
grammar or the candidate model rejects for THIS source, and the valid siblings
in the same paid response must still persist. The catch that isolates it is
narrowed to the two errors a bad proposal raises, so a defect in the composer
propagates instead of being persisted as the source's terminal refusal.
Transport faults are retried inside the LLM layer's own loop
(``tests/discovery/test_llm_transport_retry.py``).
"""

from __future__ import annotations

import asyncio
from types import SimpleNamespace
from typing import Any
from unittest.mock import AsyncMock, MagicMock, patch

import httpx
import pytest
from pydantic import BaseModel, ValidationError

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
    def __init__(
        self,
        name: str,
        source_id: str,
        *,
        error: BaseException | None = None,
    ):
        self.source_id = source_id
        self.description = "Poloidal flux at the plasma boundary"
        self.kind = "scalar"
        self.dd_paths = [source_id]
        self.reason = ""
        self.segments = _FakeSegments("ratio_of" if error is not None else name)
        self._name = name
        self._error = error

    def compose_name(self) -> str:
        if self._error is not None:
            raise self._error
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
                _FakeCandidate("", _REFUSED_PATH, error=ValueError(_GRAMMAR_MESSAGE)),
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


class _ComposedName(BaseModel):
    name: str


def _validation_error() -> ValidationError:
    """A pydantic ValidationError of the shape a bad proposal raises."""
    try:
        _ComposedName()
    except ValidationError as exc:
        return exc
    raise AssertionError("a missing field must raise ValidationError")


class TestPerCandidateCatchNarrowing:
    """Only a bad proposal is recorded; a defect in the composer propagates."""

    @pytest.mark.asyncio
    async def test_validation_error_is_recorded_for_its_own_source(
        self, compose_deps
    ) -> None:
        from imas_codex.standard_names.worker_tasks.worker_compose_batches import (
            compose_batch,
        )

        result = _FakeBatchResult(
            [
                _FakeCandidate("", _REFUSED_PATH, error=_validation_error()),
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
        assert refusal["reason"].startswith("grammar validation: ")

    @pytest.mark.asyncio
    async def test_programming_error_propagates_and_is_not_recorded(
        self, compose_deps
    ) -> None:
        """An AttributeError is this code's defect, not a source's refusal."""
        from imas_codex.standard_names.worker_tasks.worker_compose_batches import (
            compose_batch,
        )

        defect = AttributeError("segments has no base_token")
        result = _FakeBatchResult(
            [
                _FakeCandidate("", _REFUSED_PATH, error=defect),
                _FakeCandidate(_GOOD_NAME, _GOOD_PATH),
            ]
        )
        llm_out = _FakeLLMOut(result)

        async def mock_llm(*_args, **_kwargs):
            return llm_out

        with (
            patch("imas_codex.discovery.base.llm.acall_llm_structured", mock_llm),
            pytest.raises(AttributeError) as raised,
        ):
            await compose_batch(
                [_make_batch_item(_REFUSED_PATH), _make_batch_item(_GOOD_PATH)],
                _FakeBudgetManager(),
                asyncio.Event(),
            )

        assert raised.value is defect
        assert compose_deps.refusals == []
        assert compose_deps.persisted == []


class TestOneStructuredCallPerAttempt:
    """The structured call is made once per compose attempt, never in a loop.

    Transport retries belong to the LLM layer's own loop
    (``tests/discovery/test_llm_transport_retry.py``). A retry wrapper added
    here would re-spend an attempt on a fault the layer below already retried,
    and would multiply the paid calls one compose attempt makes.
    """

    @pytest.mark.asyncio
    async def test_a_successful_attempt_makes_exactly_one_call(
        self, compose_deps
    ) -> None:
        from imas_codex.standard_names.worker_tasks.worker_compose_batches import (
            compose_batch,
        )

        calls: list[int] = []

        async def answering(*_args, **_kwargs):
            calls.append(1)
            return _FakeLLMOut(_FakeBatchResult([]))

        with patch("imas_codex.discovery.base.llm.acall_llm_structured", answering):
            await compose_batch(
                [_make_batch_item(_GOOD_PATH)], _FakeBudgetManager(), asyncio.Event()
            )

        assert len(calls) == 1

    @pytest.mark.asyncio
    async def test_a_transport_fault_is_not_retried_inside_compose_batch(
        self, compose_deps
    ) -> None:
        from imas_codex.standard_names.worker_tasks.worker_compose_batches import (
            compose_batch,
        )

        calls: list[int] = []

        async def raising(*_args, **_kwargs):
            calls.append(1)
            raise httpx.ReadError("")

        with (
            patch("imas_codex.discovery.base.llm.acall_llm_structured", raising),
            pytest.raises(httpx.ReadError),
        ):
            await compose_batch(
                [_make_batch_item(_GOOD_PATH)], _FakeBudgetManager(), asyncio.Event()
            )

        assert len(calls) == 1
