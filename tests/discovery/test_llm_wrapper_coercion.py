"""Tests for tolerant batch-wrapper coercion in structured LLM parsing.

Covers ``_coerce_to_wrapper`` / ``_wrapper_field`` (pure helpers) and an
end-to-end ``acall_llm_structured`` path where the LLM returns the bare inner
object instead of the requested ``{reviews: [...]}`` wrapper — the coercion
must succeed without consuming a retry.
"""

from __future__ import annotations

from unittest.mock import AsyncMock, patch

import pytest
from pydantic import BaseModel, Field

from imas_codex.discovery.base.llm import (
    _coerce_to_wrapper,
    _parse_structured_content,
    _wrapper_field,
    acall_llm_structured,
)
from imas_codex.standard_names.models import (
    StandardNameQualityReviewDocs,
    StandardNameQualityReviewDocsBatch,
)

# ---------------------------------------------------------------------------
# Fixtures: models exercising each detection branch
# ---------------------------------------------------------------------------


class Inner(BaseModel):
    name: str
    score: float


class Wrapper(BaseModel):
    """Single required list-of-model field — the canonical wrapper."""

    items: list[Inner]


class WrapperWithOptional(BaseModel):
    """One required list field plus an optional field — still a wrapper."""

    items: list[Inner]
    note: str = ""


class WrapperWithFactory(BaseModel):
    """One required list field plus a default_factory field — still a wrapper."""

    items: list[Inner]
    extras: list[str] = Field(default_factory=list)


class MultiRequired(BaseModel):
    """Two required list fields — NOT a wrapper (ambiguous)."""

    a: list[Inner]
    b: list[Inner]


class SingleObject(BaseModel):
    """A multi-field non-list model — NOT a wrapper."""

    name: str
    score: float


class ListOfScalars(BaseModel):
    """One required list field whose element is not a model — NOT a wrapper."""

    values: list[str]


def _inner_dict() -> dict:
    return {"name": "x", "score": 1.0}


# ---------------------------------------------------------------------------
# _wrapper_field detection
# ---------------------------------------------------------------------------


def test_wrapper_field_detects_single_list_model():
    assert _wrapper_field(Wrapper) == ("items", Inner)


def test_wrapper_field_allows_optional_second_field():
    assert _wrapper_field(WrapperWithOptional) == ("items", Inner)


def test_wrapper_field_allows_default_factory_field():
    assert _wrapper_field(WrapperWithFactory) == ("items", Inner)


def test_wrapper_field_rejects_multi_required():
    assert _wrapper_field(MultiRequired) is None


def test_wrapper_field_rejects_single_object():
    assert _wrapper_field(SingleObject) is None


def test_wrapper_field_rejects_list_of_scalars():
    assert _wrapper_field(ListOfScalars) is None


def test_wrapper_field_rejects_non_model():
    assert _wrapper_field(dict) is None
    assert _wrapper_field(int) is None


def test_wrapper_field_detects_production_batch():
    field, inner = _wrapper_field(StandardNameQualityReviewDocsBatch)
    assert field == "reviews"
    assert inner is StandardNameQualityReviewDocs


# ---------------------------------------------------------------------------
# _coerce_to_wrapper behaviour
# ---------------------------------------------------------------------------


def test_coerce_bare_dict_into_wrapper():
    result = _coerce_to_wrapper(_inner_dict(), Wrapper)
    assert isinstance(result, Wrapper)
    assert len(result.items) == 1
    assert result.items[0].name == "x"


def test_coerce_list_of_dicts_into_wrapper():
    result = _coerce_to_wrapper([_inner_dict(), _inner_dict()], Wrapper)
    assert isinstance(result, Wrapper)
    assert len(result.items) == 2


def test_coerce_into_wrapper_with_optional_field():
    result = _coerce_to_wrapper(_inner_dict(), WrapperWithOptional)
    assert isinstance(result, WrapperWithOptional)
    assert len(result.items) == 1
    assert result.note == ""


def test_coerce_invalid_dict_returns_none():
    # Dict that does not validate as the inner model (missing required fields).
    assert _coerce_to_wrapper({"unrelated": 1}, Wrapper) is None


def test_coerce_multi_field_model_returns_none():
    # Not a wrapper → no-op even though the dict could be a SingleObject.
    assert _coerce_to_wrapper(_inner_dict(), SingleObject) is None


def test_coerce_already_valid_payload_is_noop_target():
    # A non-wrapper model must always coerce to None.
    assert _coerce_to_wrapper([_inner_dict()], MultiRequired) is None


def test_coerce_non_dict_non_list_returns_none():
    assert _coerce_to_wrapper("scalar", Wrapper) is None
    assert _coerce_to_wrapper(42, Wrapper) is None


def test_coerce_production_models():
    inner = {
        "source_id": "dd:foo/bar",
        "standard_name": "foo_bar",
        "scores": {
            "description_quality": 18,
            "documentation_quality": 18,
            "completeness": 18,
            "physics_accuracy": 18,
        },
        "reasoning": "ok",
        "issues": [],
    }
    result = _coerce_to_wrapper(inner, StandardNameQualityReviewDocsBatch)
    assert isinstance(result, StandardNameQualityReviewDocsBatch)
    assert len(result.reviews) == 1
    assert result.reviews[0].source_id == "dd:foo/bar"


# ---------------------------------------------------------------------------
# _parse_structured_content: retry-preserving validation wrapper
# ---------------------------------------------------------------------------


def test_parse_valid_wrapper_passthrough():
    content = '{"items": [{"name": "x", "score": 1.0}]}'
    parsed = _parse_structured_content(content, Wrapper, "test-model")
    assert isinstance(parsed, Wrapper)
    assert parsed.items[0].name == "x"


def test_parse_bare_dict_is_coerced():
    content = '{"name": "x", "score": 1.0}'
    parsed = _parse_structured_content(content, Wrapper, "test-model")
    assert isinstance(parsed, Wrapper)
    assert len(parsed.items) == 1


def test_parse_non_wrapper_invalid_raises():
    # SingleObject is not a wrapper; invalid content must raise (→ retry).
    import pydantic

    with pytest.raises(pydantic.ValidationError):
        _parse_structured_content('{"name": "x"}', SingleObject, "test-model")


def test_parse_truncated_json_raises_validation_error():
    # Truncated JSON: model_validate_json fails, json.loads fails → re-raise
    # the original ValidationError so the existing retry path handles it.
    import pydantic

    with pytest.raises(pydantic.ValidationError):
        _parse_structured_content('{"items": [{"name"', Wrapper, "test-model")


# ---------------------------------------------------------------------------
# End-to-end: bare inner JSON succeeds without consuming a retry
# ---------------------------------------------------------------------------


class _FakeUsage:
    prompt_tokens = 100
    completion_tokens = 20
    prompt_tokens_details = None


class _FakeMessage:
    def __init__(self, content: str) -> None:
        self.content = content


class _FakeChoice:
    def __init__(self, content: str) -> None:
        self.message = _FakeMessage(content)


class _FakeResponse:
    def __init__(self, content: str) -> None:
        self.choices = [_FakeChoice(content)]
        self.usage = _FakeUsage()
        self.model = "test-model"
        self._hidden_params = {"response_cost": 0.0}


async def test_acall_structured_coerces_bare_item_no_retry(monkeypatch):
    """qwen-style bare inner object → coerced, single API call, no retry."""
    monkeypatch.setenv("OPENROUTER_API_KEY_IMAS_CODEX", "sk-test")

    bare_inner = (
        '{"source_id": "dd:foo/bar", "standard_name": "foo_bar", '
        '"scores": {"description_quality": 18, "documentation_quality": 18, '
        '"completeness": 18, "physics_accuracy": 18}, '
        '"reasoning": "ok", "issues": []}'
    )
    fake = AsyncMock(return_value=_FakeResponse(bare_inner))

    with patch("litellm.acompletion", fake):
        result = await acall_llm_structured(
            model="openrouter/qwen/qwen-max",
            messages=[{"role": "user", "content": "review this"}],
            response_model=StandardNameQualityReviewDocsBatch,
            service="standard-names",
        )

    # Single API call — no retry consumed.
    assert fake.call_count == 1
    batch = result.parsed
    assert isinstance(batch, StandardNameQualityReviewDocsBatch)
    assert len(batch.reviews) == 1
    assert batch.reviews[0].source_id == "dd:foo/bar"


class TestReasoningEffortProviderShape:
    """reasoning_effort must be provider-shaped: vLLM chat_template_kwargs
    vs OpenRouter unified reasoning field."""

    @staticmethod
    def _kwargs(model: str) -> dict:
        from imas_codex.discovery.base.llm import _build_kwargs

        return _build_kwargs(
            model=model,
            messages=[{"role": "user", "content": "x"}],
            api_key=None,
            response_format=None,
            max_tokens=None,
            temperature=0.0,
            timeout=None,
            service="test",
            reasoning_effort="high",
        )

    def test_hosted_vllm_gets_chat_template_kwargs(self):
        kw = self._kwargs("hosted_vllm/deepseek-v4-flash")
        eb = kw.get("extra_body") or {}
        assert eb.get("chat_template_kwargs") == {
            "thinking": True,
            "reasoning_effort": "high",
        }
        assert "reasoning" not in eb

    def test_openrouter_keeps_unified_reasoning_field(self):
        kw = self._kwargs("openrouter/qwen/qwen3.7-max")
        eb = kw.get("extra_body") or {}
        assert eb.get("reasoning") == {"effort": "high"}
        assert "chat_template_kwargs" not in eb

    @staticmethod
    def _kwargs_effort(model: str, effort: str) -> dict:
        from imas_codex.discovery.base.llm import _build_kwargs

        return _build_kwargs(
            model=model,
            messages=[{"role": "user", "content": "x"}],
            api_key=None,
            response_format=None,
            max_tokens=None,
            temperature=0.0,
            timeout=None,
            service="test",
            reasoning_effort=effort,
        )

    def test_openrouter_max_maps_to_xhigh(self):
        # "max" is a vLLM-only level; OpenRouter rejects it (enum tops at
        # "xhigh"), so cross-provider "max" must be mapped to the OpenRouter max.
        kw = self._kwargs_effort("openrouter/moonshotai/kimi-k2.6", "max")
        eb = kw.get("extra_body") or {}
        assert eb.get("reasoning") == {"effort": "xhigh"}

    def test_hosted_vllm_max_preserved(self):
        # Local vLLM accepts "max" verbatim — it must not be remapped.
        kw = self._kwargs_effort("hosted_vllm/deepseek-v4-flash", "max")
        eb = kw.get("extra_body") or {}
        assert eb.get("chat_template_kwargs") == {
            "thinking": True,
            "reasoning_effort": "max",
        }


# ---------------------------------------------------------------------------
# Empty-content responses are retryable (reasoning-budget exhaustion)
# ---------------------------------------------------------------------------
#
# A reasoning model in thinking mode (DeepSeek-V4 at reasoning_effort="max")
# can spend its entire completion-token budget on reasoning and return
# finish_reason="length" with EMPTY content. Before the fix this raised a
# non-retryable ValueError; now it must retry (and grow the token budget on
# the length case) so a single transient exhaustion does not kill a compose
# item.


class _UsageFixed:
    prompt_tokens = 120
    completion_tokens = 30
    prompt_tokens_details = None


class _ChoiceFR:
    """Choice carrying a finish_reason, like the real OpenAI/litellm shape."""

    def __init__(self, content: str | None, finish_reason: str) -> None:
        self.message = _FakeMessage(content)
        self.finish_reason = finish_reason


class _ResponseFR:
    def __init__(self, content: str | None, finish_reason: str) -> None:
        self.choices = [_ChoiceFR(content, finish_reason)]
        self.usage = _UsageFixed()
        self.model = "test-model"
        self._hidden_params = {"response_cost": 0.0}


_VALID_REVIEW_JSON = (
    '{"reviews": [{"source_id": "dd:foo/bar", "standard_name": "foo_bar", '
    '"scores": {"description_quality": 18, "documentation_quality": 18, '
    '"completeness": 18, "physics_accuracy": 18}, '
    '"reasoning": "ok", "issues": []}]}'
)


def test_empty_response_error_is_retryable():
    """The empty-content sentinel message lands in the retry path."""
    from imas_codex.discovery.base.llm import EmptyResponseError, _is_retryable

    err = EmptyResponseError("length")
    assert _is_retryable(str(err)) is True
    assert err.finish_reason == "length"


def test_upstream_500_is_retryable_but_not_rate_limited():
    """Provider 500s retry the single call but never pull back concurrency.

    A 500 is a backend transient, distinct from a 429: retrying recovers it,
    whereas the AIMD governor's concurrency pullback (429-only) would not.
    """
    from imas_codex.discovery.base.llm import _is_rate_limited, _is_retryable

    openrouter_500 = (
        "litellm.APIError: APIError: OpenrouterException - The server had an "
        "error processing your request. Sorry about that!"
    )
    internal_500 = (
        "litellm.InternalServerError: Internal server error from provider"
    )
    for msg in (openrouter_500, internal_500):
        assert _is_retryable(msg) is True
        # A 500 must NOT be misread as a rate-limit — it stays out of the
        # governor's backoff path.
        assert _is_rate_limited(msg) is False


def test_bump_max_tokens_grows_then_caps():
    """The length-retry budget bump doubles up to a hard cap, then stops."""
    from imas_codex.discovery.base.llm import (
        _LENGTH_RETRY_TOKEN_CAP,
        _bump_max_tokens_for_length,
    )

    # Cap-agnostic (survives cap re-sizing): a budget one doubling below the cap
    # grows once, clamps to the cap, then stops bumping.
    start = _LENGTH_RETRY_TOKEN_CAP // 2
    kwargs = {"max_tokens": start}
    assert _bump_max_tokens_for_length(kwargs) == _LENGTH_RETRY_TOKEN_CAP
    assert kwargs["max_tokens"] == _LENGTH_RETRY_TOKEN_CAP
    # At the cap, no further bump is applied.
    assert _bump_max_tokens_for_length(kwargs) is None
    assert kwargs["max_tokens"] == _LENGTH_RETRY_TOKEN_CAP

    # A value just below the cap doubles but clamps to the cap, never overshoots.
    kwargs2 = {"max_tokens": _LENGTH_RETRY_TOKEN_CAP - 1}
    assert _bump_max_tokens_for_length(kwargs2) == _LENGTH_RETRY_TOKEN_CAP


async def test_empty_length_response_retries_and_bumps_budget(monkeypatch):
    """finish_reason='length' empty → retry succeeds AND the budget grows."""
    monkeypatch.setenv("OPENROUTER_API_KEY_IMAS_CODEX", "sk-test")

    seen_max_tokens: list[int | None] = []

    async def fake(**kwargs):
        seen_max_tokens.append(kwargs.get("max_tokens"))
        if len(seen_max_tokens) == 1:
            # First attempt: reasoning ate the whole budget → empty + length.
            return _ResponseFR(None, "length")
        return _ResponseFR(_VALID_REVIEW_JSON, "stop")

    with patch("litellm.acompletion", fake):
        result = await acall_llm_structured(
            model="openrouter/qwen/qwen-max",
            messages=[{"role": "user", "content": "review this"}],
            response_model=StandardNameQualityReviewDocsBatch,
            service="standard-names",
            retry_base_delay=0.0,  # no real sleep in the test
        )

    # Two API calls: the empty-length response was retried, not hard-failed.
    assert len(seen_max_tokens) == 2
    # The retry ran with a larger token budget than the first attempt.
    assert seen_max_tokens[1] is not None
    assert seen_max_tokens[0] is not None
    assert seen_max_tokens[1] > seen_max_tokens[0]
    batch = result.parsed
    assert isinstance(batch, StandardNameQualityReviewDocsBatch)
    assert batch.reviews[0].source_id == "dd:foo/bar"


async def test_empty_nonlength_response_retries_without_bump(monkeypatch):
    """A transient empty that is NOT length-exhaustion still retries, but the
    token budget is left unchanged (no spurious growth)."""
    monkeypatch.setenv("OPENROUTER_API_KEY_IMAS_CODEX", "sk-test")

    seen_max_tokens: list[int | None] = []

    async def fake(**kwargs):
        seen_max_tokens.append(kwargs.get("max_tokens"))
        if len(seen_max_tokens) == 1:
            return _ResponseFR(None, "stop")  # empty but not budget-exhausted
        return _ResponseFR(_VALID_REVIEW_JSON, "stop")

    with patch("litellm.acompletion", fake):
        await acall_llm_structured(
            model="openrouter/qwen/qwen-max",
            messages=[{"role": "user", "content": "review this"}],
            response_model=StandardNameQualityReviewDocsBatch,
            service="standard-names",
            retry_base_delay=0.0,
        )

    assert len(seen_max_tokens) == 2
    # No length-exhaustion → budget unchanged across the retry.
    assert seen_max_tokens[0] == seen_max_tokens[1]
