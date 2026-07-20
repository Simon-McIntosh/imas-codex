"""Tests for _build_kwargs JSON schema conversion and _sanitize_content in llm.py."""

from __future__ import annotations

import pytest
from pydantic import BaseModel, Field

from imas_codex.discovery.base.llm import (
    _ensure_json_in_messages,
    _extract_balanced_json,
    _find_json_start,
    _is_pydantic_model,
    _messages_mention_json,
    _sanitize_content,
    _strip_unsupported_schema_props,
    _to_json_schema_format,
)


class DummyResponse(BaseModel):
    """Simple model for testing."""

    name: str
    score: float


class NestedResponse(BaseModel):
    """Model with nested types for testing."""

    items: list[str]
    metadata: dict[str, str]


def test_is_pydantic_model_class():
    assert _is_pydantic_model(DummyResponse) is True


def test_is_pydantic_model_rejects_instance():
    assert _is_pydantic_model(DummyResponse(name="x", score=1.0)) is False


def test_is_pydantic_model_rejects_dict():
    assert _is_pydantic_model({"type": "json_object"}) is False


def test_is_pydantic_model_rejects_none():
    assert _is_pydantic_model(None) is False


def test_to_json_schema_format_structure():
    result = _to_json_schema_format(DummyResponse)
    assert result["type"] == "json_schema"
    assert result["json_schema"]["name"] == "DummyResponse"
    assert result["json_schema"]["strict"] is False
    schema = result["json_schema"]["schema"]
    assert "properties" in schema
    assert "name" in schema["properties"]
    assert "score" in schema["properties"]


def test_to_json_schema_format_nested():
    result = _to_json_schema_format(NestedResponse)
    schema = result["json_schema"]["schema"]
    assert "items" in schema["properties"]
    assert "metadata" in schema["properties"]


class ModelWithConstraints(BaseModel):
    """Model with array/string constraints that some providers reject."""

    keywords: list[str] = Field(default_factory=list, max_length=8)
    name: str = Field(min_length=1, max_length=100)


def test_strip_unsupported_schema_props_removes_maxItems():
    """maxItems from max_length on list fields must be stripped."""
    schema = ModelWithConstraints.model_json_schema()
    cleaned = _strip_unsupported_schema_props(schema)
    kw_prop = cleaned["properties"]["keywords"]
    assert "maxItems" not in kw_prop


def test_to_json_schema_format_strips_constraints():
    """_to_json_schema_format should produce a provider-safe schema."""
    result = _to_json_schema_format(ModelWithConstraints)
    schema = result["json_schema"]["schema"]
    kw_prop = schema["properties"]["keywords"]
    assert "maxItems" not in kw_prop
    # Pydantic validation still enforces it — just not in the API schema


@pytest.mark.parametrize(
    "model_name",
    [
        "anthropic/claude-sonnet-4.6",
        "openrouter/anthropic/claude-sonnet-4.6",
        "google/gemini-3-flash-preview",
        "openai/gpt-5.4",
        "gpt-5.2-codex",
    ],
)
def test_build_kwargs_wraps_pydantic_for_all_models(model_name, monkeypatch):
    """All models get Pydantic→json_schema conversion, not just GPT-5."""
    from imas_codex.discovery.base import llm

    # Stub out settings/proxy lookups
    monkeypatch.setattr(llm, "get_llm_location", lambda: "local", raising=False)
    monkeypatch.setattr(llm, "get_llm_proxy_url", lambda: None, raising=False)

    kwargs = llm._build_kwargs(
        model=model_name,
        api_key="test-key",
        messages=[{"role": "user", "content": "test"}],
        response_format=DummyResponse,
        max_tokens=None,
        temperature=None,
        timeout=None,
    )
    rf = kwargs["response_format"]
    assert isinstance(rf, dict), f"Expected dict, got {type(rf)} for model {model_name}"
    assert rf["type"] == "json_schema"
    assert rf["json_schema"]["name"] == "DummyResponse"
    assert rf["json_schema"]["strict"] is False


def test_build_kwargs_passes_dict_response_format_unchanged(monkeypatch):
    """Non-Pydantic response_format (e.g., raw dict) is passed through."""
    from imas_codex.discovery.base import llm

    monkeypatch.setattr(llm, "get_llm_location", lambda: "local", raising=False)
    monkeypatch.setattr(llm, "get_llm_proxy_url", lambda: None, raising=False)

    raw_format = {"type": "json_object"}
    kwargs = llm._build_kwargs(
        model="anthropic/claude-sonnet-4.6",
        api_key="test-key",
        messages=[{"role": "user", "content": "test"}],
        response_format=raw_format,
        max_tokens=None,
        temperature=None,
        timeout=None,
    )
    assert kwargs["response_format"] is raw_format


# ---------------------------------------------------------------------------
# _sanitize_content tests — prose extraction, code fences, edge cases
# ---------------------------------------------------------------------------


class TestSanitizeContentBasic:
    """Basic sanitization: code fences, control chars, surrogates."""

    def test_strips_json_code_fence(self):
        raw = '```json\n{"a": 1}\n```'
        assert _sanitize_content(raw) == '{"a": 1}'

    def test_strips_bare_code_fence(self):
        raw = '```\n{"a": 1}\n```'
        assert _sanitize_content(raw) == '{"a": 1}'

    def test_removes_control_chars(self):
        raw = '{"a":\x00 1}'
        assert _sanitize_content(raw) == '{"a": 1}'

    def test_passthrough_clean_json(self):
        raw = '{"candidates": [], "vocab_gaps": []}'
        assert _sanitize_content(raw) == raw


class TestSanitizeContentProseExtraction:
    """Extract JSON from LLM prose wrappers."""

    def test_extracts_json_after_thinking_text(self):
        raw = 'Looking at these paths, I need to analyze them.\n\n{"candidates": [], "vocab_gaps": []}'
        result = _sanitize_content(raw)
        assert result == '{"candidates": [], "vocab_gaps": []}'

    def test_extracts_json_with_multiline_preamble(self):
        raw = (
            "I'll analyze the DD paths for standard names.\n"
            "First, let me consider the physics.\n"
            "Here are the results:\n"
            '{"candidates": [{"name": "electron_temperature"}], "vocab_gaps": []}'
        )
        result = _sanitize_content(raw)
        assert '"candidates"' in result
        assert result.startswith("{")

    def test_extracts_json_with_trailing_text(self):
        raw = 'Here is the output:\n{"a": 1, "b": 2}\nI hope this helps!'
        result = _sanitize_content(raw)
        assert result == '{"a": 1, "b": 2}'

    def test_extracts_nested_json(self):
        raw = 'Analysis:\n{"outer": {"inner": [1, 2, 3]}, "x": "y"}'
        result = _sanitize_content(raw)
        assert result == '{"outer": {"inner": [1, 2, 3]}, "x": "y"}'

    def test_handles_json_with_strings_containing_braces(self):
        raw = 'Result:\n{"desc": "function f(x) { return x; }", "n": 1}'
        result = _sanitize_content(raw)
        assert result == '{"desc": "function f(x) { return x; }", "n": 1}'

    def test_extracts_array_from_prose(self):
        raw = "Here are the items:\n[1, 2, 3]"
        result = _sanitize_content(raw)
        assert result == "[1, 2, 3]"

    def test_real_world_compose_error_pattern(self):
        """Reproduce the actual error pattern from SN compose failures."""
        raw = (
            "Looking at these two paths from the mhd IDS, I need to generate "
            "standard names following the grammar rules.\n\n"
            "{\n"
            '  "candidates": [\n'
            "    {\n"
            '      "name": "mhd_frequency_linear_toroidal_mode_number",\n'
            '      "description": "Toroidal mode number for MHD instability"\n'
            "    }\n"
            "  ],\n"
            '  "attachments": [],\n'
            '  "skipped": [],\n'
            '  "vocab_gaps": []\n'
            "}"
        )
        result = _sanitize_content(raw)
        assert result.startswith("{")
        assert result.endswith("}")
        import json

        parsed = json.loads(result)
        assert "candidates" in parsed


class TestFindJsonStart:
    """Unit tests for _find_json_start."""

    def test_finds_object_start(self):
        assert _find_json_start('hello\n{"a": 1}') >= 0

    def test_finds_array_start(self):
        assert _find_json_start("text [1,2]") == 5

    def test_returns_negative_for_no_json(self):
        assert _find_json_start("just plain text") == -1

    def test_skips_to_first_brace(self):
        text = 'preamble {"key": "value"}'
        idx = _find_json_start(text)
        assert text[idx] == "{"


class TestExtractBalancedJson:
    """Unit tests for _extract_balanced_json."""

    def test_simple_object(self):
        text = '{"a": 1} trailing'
        assert _extract_balanced_json(text, 0) == '{"a": 1}'

    def test_nested_object(self):
        text = '{"a": {"b": 2}} end'
        assert _extract_balanced_json(text, 0) == '{"a": {"b": 2}}'

    def test_with_string_containing_braces(self):
        text = '{"code": "if (x) { y; }"} extra'
        assert _extract_balanced_json(text, 0) == '{"code": "if (x) { y; }"}'

    def test_escaped_quotes_in_string(self):
        text = r'{"a": "say \"hello\""} more'
        assert _extract_balanced_json(text, 0) == r'{"a": "say \"hello\""}'

    def test_array(self):
        text = "[1, [2, 3], 4] rest"
        assert _extract_balanced_json(text, 0) == "[1, [2, 3], 4]"

    def test_no_close_returns_to_end(self):
        text = '{"unbalanced'
        assert _extract_balanced_json(text, 0) == '{"unbalanced'


class TestBuildKwargsServiceTag:
    """Tests for service= parameter in _build_kwargs."""

    def _build(self, monkeypatch, service="untagged", **overrides):
        from imas_codex.discovery.base import llm

        monkeypatch.setattr(llm, "get_llm_location", lambda: "local", raising=False)
        monkeypatch.setattr(llm, "get_llm_proxy_url", lambda: None, raising=False)
        defaults = {
            "model": "anthropic/claude-sonnet-4.6",
            "api_key": "test-key",
            "messages": [{"role": "user", "content": "test"}],
            "response_format": None,
            "max_tokens": None,
            "temperature": None,
            "timeout": None,
        }
        defaults.update(overrides)
        return llm._build_kwargs(**defaults, service=service)

    def test_default_service_is_untagged(self, monkeypatch):
        kwargs = self._build(monkeypatch)
        assert kwargs["metadata"]["service"] == "untagged"
        assert kwargs["extra_headers"]["X-Title"] == "imas-codex:untagged"

    def test_service_sets_metadata(self, monkeypatch):
        kwargs = self._build(monkeypatch, service="facility-discovery")
        assert kwargs["metadata"]["service"] == "facility-discovery"

    def test_service_sets_xtitle(self, monkeypatch):
        kwargs = self._build(monkeypatch, service="data-dictionary")
        assert kwargs["extra_headers"]["X-Title"] == "imas-codex:data-dictionary"

    @pytest.mark.parametrize(
        "service",
        [
            "facility-discovery",
            "standard-names",
            "data-dictionary",
            "imas-mapping",
            "embedding",
            "untagged",
        ],
    )
    def test_all_valid_services_produce_correct_headers(self, monkeypatch, service):
        kwargs = self._build(monkeypatch, service=service)
        assert kwargs["extra_headers"]["X-Title"] == f"imas-codex:{service}"
        assert kwargs["metadata"]["service"] == service

    def test_http_referer_always_present(self, monkeypatch):
        kwargs = self._build(monkeypatch, service="imas-mapping")
        assert "HTTP-Referer" in kwargs["extra_headers"]
        assert "imas-codex" in kwargs["extra_headers"]["HTTP-Referer"]


# ---------------------------------------------------------------------------
# Dashscope (qwen) "json" guard — _ensure_json_in_messages + _build_kwargs wiring
# ---------------------------------------------------------------------------


def _text_of(messages):
    """Flatten all message content (string + block form) to one lowercase string."""
    parts = []
    for m in messages:
        content = m["content"]
        if isinstance(content, str):
            parts.append(content)
        elif isinstance(content, list):
            parts.extend(str(b.get("text", "")) for b in content if isinstance(b, dict))
    return " ".join(parts).lower()


class TestEnsureJsonInMessages:
    """Unit tests for the Dashscope json-guard injection helper."""

    def test_injects_when_no_json_present(self):
        msgs = [
            {"role": "system", "content": "You are precise."},
            {"role": "user", "content": "How many apples?"},
        ]
        out = _ensure_json_in_messages(msgs)
        assert "json" in _text_of(out)

    def test_idempotent_when_json_already_present_user(self):
        msgs = [
            {"role": "system", "content": "You are precise."},
            {"role": "user", "content": "Return a JSON object with the count."},
        ]
        out = _ensure_json_in_messages(msgs)
        # No injection — text unchanged (no duplicate sentence appended).
        assert out is msgs or _text_of(out) == _text_of(msgs)
        assert _text_of(out).count("json") == 1

    def test_idempotent_when_json_already_present_system(self):
        msgs = [
            {"role": "system", "content": "Respond as json."},
            {"role": "user", "content": "How many apples?"},
        ]
        out = _ensure_json_in_messages(msgs)
        assert _text_of(out).count("json") == 1

    def test_no_double_injection_on_repeat_call(self):
        msgs = [
            {"role": "system", "content": "You are precise."},
            {"role": "user", "content": "How many apples?"},
        ]
        once = _ensure_json_in_messages(msgs)
        twice = _ensure_json_in_messages(once)
        assert _text_of(twice).count("json") == 1

    def test_appends_to_system_not_user(self):
        msgs = [
            {"role": "system", "content": "You are precise."},
            {"role": "user", "content": "How many apples?"},
        ]
        out = _ensure_json_in_messages(msgs)
        assert "json" in out[0]["content"].lower()
        assert out[1]["content"] == "How many apples?"

    def test_prepends_system_when_none_exists(self):
        msgs = [{"role": "user", "content": "How many apples?"}]
        out = _ensure_json_in_messages(msgs)
        assert out[0]["role"] == "system"
        assert "json" in out[0]["content"].lower()
        # Original user message preserved at the end.
        assert out[-1]["content"] == "How many apples?"

    def test_does_not_mutate_input(self):
        msgs = [
            {"role": "system", "content": "You are precise."},
            {"role": "user", "content": "How many apples?"},
        ]
        before = msgs[0]["content"]
        _ensure_json_in_messages(msgs)
        assert msgs[0]["content"] == before

    def test_empty_messages_returns_empty(self):
        assert _ensure_json_in_messages([]) == []

    def test_preserves_cache_control_breakpoint(self):
        """Injection into block-form system message keeps cache_control intact."""
        msgs = [
            {
                "role": "system",
                "content": [
                    {
                        "type": "text",
                        "text": "You are precise.",
                        "cache_control": {"type": "ephemeral"},
                    }
                ],
            },
            {"role": "user", "content": "How many apples?"},
        ]
        out = _ensure_json_in_messages(msgs)
        block = out[0]["content"][-1]
        assert block["cache_control"] == {"type": "ephemeral"}
        assert "json" in block["text"].lower()
        # The cacheable prefix is the same block we appended to (last block).
        assert block["text"].startswith("You are precise.")

    def test_appends_to_last_text_block_when_multiple(self):
        msgs = [
            {
                "role": "system",
                "content": [
                    {"type": "text", "text": "Block one."},
                    {
                        "type": "text",
                        "text": "Block two.",
                        "cache_control": {"type": "ephemeral"},
                    },
                ],
            },
        ]
        out = _ensure_json_in_messages(msgs)
        blocks = out[0]["content"]
        assert "json" in blocks[-1]["text"].lower()
        assert blocks[-1]["cache_control"] == {"type": "ephemeral"}
        # First block untouched (preserves more of the cacheable prefix).
        assert "json" not in blocks[0]["text"].lower()


class TestMessagesMentionJson:
    def test_detects_in_user_string(self):
        assert _messages_mention_json([{"role": "user", "content": "give JSON"}])

    def test_detects_in_block_form(self):
        msgs = [
            {
                "role": "system",
                "content": [{"type": "text", "text": "Return json please"}],
            }
        ]
        assert _messages_mention_json(msgs)

    def test_negative_when_absent(self):
        assert not _messages_mention_json([{"role": "user", "content": "hello"}])


class TestBuildKwargsJsonGuard:
    """_build_kwargs must wire json injection only when response_format is set."""

    def _build(self, monkeypatch, **overrides):
        from imas_codex.discovery.base import llm

        monkeypatch.setattr(llm, "get_llm_location", lambda: "local", raising=False)
        monkeypatch.setattr(llm, "get_llm_proxy_url", lambda: None, raising=False)
        defaults = {
            "model": "openrouter/qwen/qwen3.7-max",
            "api_key": "test-key",
            "messages": [
                {"role": "system", "content": "You are precise."},
                {"role": "user", "content": "How many apples?"},
            ],
            "response_format": DummyResponse,
            "max_tokens": None,
            "temperature": None,
            "timeout": None,
        }
        defaults.update(overrides)
        return llm._build_kwargs(**defaults)

    def test_injects_json_when_response_format_set(self, monkeypatch):
        kwargs = self._build(monkeypatch)
        assert "json" in _text_of(kwargs["messages"])

    def test_no_injection_when_response_format_none(self, monkeypatch):
        kwargs = self._build(monkeypatch, response_format=None)
        assert "json" not in _text_of(kwargs["messages"])

    def test_idempotent_when_prompt_already_mentions_json(self, monkeypatch):
        msgs = [
            {"role": "system", "content": "You are precise."},
            {"role": "user", "content": "Return a json object."},
        ]
        kwargs = self._build(monkeypatch, messages=msgs)
        assert _text_of(kwargs["messages"]).count("json") == 1

    def test_cache_control_preserved_for_cache_capable_model(self, monkeypatch):
        """Anthropic models get cache_control; injection must not break it."""
        kwargs = self._build(
            monkeypatch,
            model="openrouter/anthropic/claude-sonnet-4.6",
        )
        sys_content = kwargs["messages"][0]["content"]
        # inject_cache_control converts the system message to block form.
        assert isinstance(sys_content, list)
        last_block = sys_content[-1]
        assert last_block.get("cache_control") == {"type": "ephemeral"}
        assert "json" in _text_of(kwargs["messages"])
