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
