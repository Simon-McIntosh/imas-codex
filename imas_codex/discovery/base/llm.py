"""
Common LLM infrastructure for discovery operations.

Provides a uniform interface for calling LLMs across discovery domains
(paths, wiki, signals). Handles:
- LiteLLM noise suppression (stdout/stderr print messages)
- API key management
- Retry with exponential backoff (wraps both LLM call + response parsing)
- Cost extraction and accumulation across retries
- JSON sanitization and Pydantic structured output parsing
- Model-aware token limits

Both sync and async entry points share identical retry/parse/cost logic.
The retry loop wraps both the API call and the response parsing so that
truncated JSON or Pydantic validation errors from malformed responses
trigger a fresh LLM attempt rather than an immediate failure.

Usage:
    from imas_codex.discovery.base.llm import (
        call_llm_structured,
        acall_llm_structured,
        call_llm,
        acall_llm,
    )

    # Structured output with retry+parse (preferred for scoring)
    batch, cost, tokens = call_llm_structured(
        model="google/gemini-3-flash-preview",
        messages=[...],
        response_model=ScoreBatch,
    )

    # Async structured output (also returns LLMResult with cache info)
    llm_out = await acall_llm_structured(
        model="google/gemini-3-flash-preview",
        messages=[...],
        response_model=WikiScoreBatch,
    )
    batch, cost, tokens = llm_out          # backward-compatible
    cache_read = llm_out.cache_read_tokens  # new: cache metrics

    # Raw response (when caller needs custom parsing)
    response, cost = call_llm(
        model="google/gemini-3-flash-preview",
        messages=[...],
    )
"""

from __future__ import annotations

import asyncio
import json
import logging
import os
import re
import time
from functools import lru_cache
from pathlib import Path
from typing import TYPE_CHECKING, Any, Literal, TypeVar, get_args, get_origin

import yaml
from pydantic import BaseModel, ValidationError

if TYPE_CHECKING:
    pass

logger = logging.getLogger(__name__)

T = TypeVar("T")

LLM_SERVICE = Literal[
    "facility-discovery",
    "standard-names",
    "data-dictionary",
    "imas-mapping",
    "embedding",
    "rate-probe",
    "untagged",
]

_VALID_SERVICES: set[str] = set(LLM_SERVICE.__args__)  # type: ignore[attr-defined]


# ---------------------------------------------------------------------------
# Fatal error: provider budget/credit exhaustion
# ---------------------------------------------------------------------------


class ProviderBudgetExhausted(Exception):
    """Raised when the LLM provider rejects requests due to credit/budget limits.

    This is a fatal, non-retryable error. Callers should halt all LLM-dependent
    work immediately rather than retrying — the key/account needs manual
    intervention (top-up credits or raise the spending limit).
    """


class LLMResult:
    """Return type for call_llm_structured / acall_llm_structured.

    Backward-compatible with 3-tuple unpacking::

        result, cost, tokens = call_llm_structured(...)  # still works

    Also carries prompt-cache token counts for callers that need them::

        llm_out = await acall_llm_structured(...)
        result, cost, tokens = llm_out
        cache_read = llm_out.cache_read_tokens
        cache_creation = llm_out.cache_creation_tokens

    Attributes:
        parsed: The Pydantic model instance returned by the LLM.
        cost: Total cost in USD (accumulated across retries).
        tokens: Total tokens (prompt + completion).
        cache_read_tokens: Tokens served from provider prompt cache (0 if
            the provider doesn't report caching or the prompt wasn't cached).
        cache_creation_tokens: Tokens written to the provider prompt cache.
    """

    __slots__ = (
        "parsed",
        "cost",
        "tokens",
        "input_tokens",
        "output_tokens",
        "cache_read_tokens",
        "cache_creation_tokens",
    )

    def __init__(
        self,
        parsed: Any,
        cost: float,
        tokens: int,
        cache_read_tokens: int = 0,
        cache_creation_tokens: int = 0,
        input_tokens: int = 0,
        output_tokens: int = 0,
    ) -> None:
        self.parsed = parsed
        self.cost = cost
        self.tokens = tokens
        # Split prompt/completion counts. When the caller doesn't provide
        # them (backward-compat), default to 0 — consumers that need the
        # split should read these fields rather than unpacking the tuple.
        self.input_tokens = input_tokens
        self.output_tokens = output_tokens
        self.cache_read_tokens = cache_read_tokens
        self.cache_creation_tokens = cache_creation_tokens

    # Allow ``result, cost, tokens = call_llm_structured(...)``
    def __iter__(self):
        return iter((self.parsed, self.cost, self.tokens))

    def __len__(self) -> int:
        return 3

    def __repr__(self) -> str:
        return (
            f"LLMResult(cost={self.cost:.4f}, tokens={self.tokens}, "
            f"cache_read={self.cache_read_tokens}, "
            f"cache_creation={self.cache_creation_tokens})"
        )


# Patterns indicating the API key or account has hit a hard spending cap.
# Matched case-insensitively against the full error message.
_BUDGET_EXHAUSTED_PATTERNS = (
    "requires more credits",
    "insufficient_quota",
    "billing_hard_limit_reached",
    "exceeded your current quota",
    "payment required",
)


def _is_budget_exhausted(error_msg: str) -> bool:
    """Return True if *error_msg* indicates a hard credit/budget limit."""
    msg = error_msg.lower()
    # HTTP 402 Payment Required is the canonical signal from OpenRouter
    if "402" in msg and ("credit" in msg or "payment" in msg or "afford" in msg):
        return True
    return any(p in msg for p in _BUDGET_EXHAUSTED_PATTERNS)


# ---------------------------------------------------------------------------
# Retry configuration
# ---------------------------------------------------------------------------
DEFAULT_MAX_RETRIES = 5
DEFAULT_RETRY_BASE_DELAY = 5.0  # seconds, doubles each retry

# Error patterns that warrant retry (all lowercase for matching).
# Proven in wiki pipeline — shared across all discovery domains.
RETRYABLE_PATTERNS = frozenset(
    {
        "overloaded",
        "rate",
        "429",
        "503",
        "timeout",
        "eof",  # EOF while parsing JSON (truncated response)
        "json",  # JSON parsing errors
        "truncated",
        "validation",  # Pydantic validation errors from malformed responses
    }
)

# Errors that should NOT be retried even though they match a retryable
# pattern (e.g. "validation").  Vocab-gap errors are deterministic — the LLM
# will keep selecting the same unregistered token on every attempt.
NON_RETRYABLE_PATTERNS = frozenset(
    {
        "not a registered",  # IR segment vocab-gap validation
        "kind must be one of",  # schema enum violation (deterministic)
    }
)

# ---------------------------------------------------------------------------
# Model-aware token limits
# ---------------------------------------------------------------------------
# Gemini 3 Flash: 1M context, 65k output, $0.10/$0.40 per 1M tokens
# Claude Sonnet: 200k context, ~8k output default
# Claude Haiku: 200k context, ~4k output default
#
# These limits are intentionally generous for Gemini Flash since it's
# the primary scoring model and large batches (50+ items) need room.
MODEL_TOKEN_LIMITS: dict[str, dict[str, int]] = {
    "gemini": {
        "max_tokens": 16000,  # Observed completions are ~9-10K; 16K gives ample headroom
        "timeout": 120,  # 2 min — large batches take time
    },
    "claude": {
        "max_tokens": 32000,
        "timeout": 120,
    },
    "hosted_vllm": {
        "max_tokens": 32000,
        "timeout": 300,  # 5 min — local GPU with high concurrency needs headroom
    },
    "default": {
        "max_tokens": 32000,
        "timeout": 120,
    },
}


def get_model_limits(model: str) -> dict[str, int]:
    """Get token limits for a model based on its family.

    Args:
        model: Model identifier (e.g., "google/gemini-3-flash-preview")

    Returns:
        Dict with max_tokens and timeout values.
    """
    model_lower = model.lower()
    for family, limits in MODEL_TOKEN_LIMITS.items():
        if family in model_lower:
            return limits
    return MODEL_TOKEN_LIMITS["default"]


# ---------------------------------------------------------------------------
# LiteLLM noise suppression
# ---------------------------------------------------------------------------


@lru_cache(maxsize=1)
def _can_reach_github(timeout: float = 1.5) -> bool:
    """Fast cached check for GitHub raw content connectivity.

    Used to decide whether LiteLLM should fetch remote model pricing
    and Anthropic beta headers, or use bundled local copies.

    Returns True when GitHub is reachable (e.g. login nodes with
    internet), False on air-gapped compute nodes (e.g. Titan).
    """
    import socket

    try:
        socket.create_connection(
            ("raw.githubusercontent.com", 443), timeout=timeout
        ).close()
        return True
    except OSError:
        return False


def set_litellm_offline_env() -> None:
    """Set env vars to prevent LiteLLM import-time remote fetches.

    **Must be called before** ``import litellm`` to be effective.
    Only sets the vars when GitHub is unreachable (air-gapped nodes).
    When GitHub is reachable, lets LiteLLM fetch the latest data.
    """
    if not _can_reach_github():
        os.environ.setdefault("LITELLM_LOCAL_MODEL_COST_MAP", "True")
        os.environ.setdefault("LITELLM_LOCAL_ANTHROPIC_BETA_HEADERS", "True")
        logger.debug("Air-gapped: using local LiteLLM model cost map")


def suppress_litellm_noise() -> None:
    """Suppress all LiteLLM and HuggingFace diagnostic output.

    LiteLLM prints "Give Feedback", "Provider List", and debug info
    directly to stdout/stderr via print() calls, bypassing Python's
    logging system. This function suppresses both:
    1. Logger-based output (via logging levels)
    2. Print-based output (via litellm.suppress_debug_info)

    Also suppresses huggingface_hub/transformers/sentence_transformers
    logging which pollutes discovery output when these are installed.

    Call this once at module load time in any module that uses LiteLLM.
    """
    set_litellm_offline_env()
    try:
        import litellm
    except ModuleNotFoundError:
        # litellm not installed — nothing to suppress.  This can happen
        # when the CLI is imported outside the managed venv (e.g. bare
        # ``python`` on WSL) and litellm is not on sys.path.
        return

    # Suppress print-based diagnostic messages
    litellm.suppress_debug_info = True

    # Suppress all litellm logging to ERROR level
    for logger_name in (
        "LiteLLM",
        "LiteLLM Proxy",
        "LiteLLM Router",
        "httpx",
        "huggingface_hub",
        "sentence_transformers",
        "transformers",
    ):
        level = logging.WARNING if logger_name == "httpx" else logging.ERROR
        logging.getLogger(logger_name).setLevel(level)

    # Environment variables for litellm internals
    os.environ.setdefault("LITELLM_LOG", "ERROR")
    # Disable hf_xet native bindings (set early in __init__.py too,
    # but reinforce here in case suppress_litellm_noise is called
    # before package __init__ in some import path)
    os.environ.setdefault("HF_HUB_DISABLE_XET", "1")


# ---------------------------------------------------------------------------
# API key / model helpers
# ---------------------------------------------------------------------------


def get_api_key() -> str:
    """Get OpenRouter API key from environment.

    Raises:
        ValueError: If OPENROUTER_API_KEY_IMAS_CODEX is not set.
    """
    api_key = os.environ.get("OPENROUTER_API_KEY_IMAS_CODEX")
    if not api_key:
        raise ValueError(
            "OPENROUTER_API_KEY_IMAS_CODEX not set. Add to .env or export."
        )
    return api_key


def get_api_key_for_service(service: str) -> str:
    """Get OpenRouter API key, with per-service override support.

    Checks OPENROUTER_API_KEY_<SERVICE_UPPER> first (hyphens → underscores),
    then falls back to OPENROUTER_API_KEY_IMAS_CODEX.

    Examples:
        service="facility-discovery" → checks OPENROUTER_API_KEY_FACILITY_DISCOVERY
        service="standard-names" → checks OPENROUTER_API_KEY_STANDARD_NAMES
        service="untagged" → checks OPENROUTER_API_KEY_UNTAGGED (unlikely set)
    """
    if service and service != "untagged":
        env_var = f"OPENROUTER_API_KEY_{service.upper().replace('-', '_')}"
        per_service_key = os.environ.get(env_var)
        if per_service_key:
            return per_service_key
    return get_api_key()


_LOCAL_MODEL_PREFIXES = ("ollama/", "hosted_vllm/", "openai/localhost")


def ensure_model_prefix(model: str) -> str:
    """Ensure model ID has the correct provider prefix for LiteLLM routing.

    OpenRouter models get the ``openrouter/`` prefix to preserve
    ``cache_control`` blocks. Local models (ollama, vLLM) are passed
    through without modification.
    """
    if any(model.startswith(p) for p in _LOCAL_MODEL_PREFIXES):
        return model
    if not model.startswith("openrouter/"):
        return f"openrouter/{model}"
    return model


_PREFIX_WARNED: set[str] = set()


def _warn_if_missing_openrouter_prefix(model: str) -> None:
    """Warn once per model if it lacks ``openrouter/`` and a direct key is set.

    Unprefixed model IDs silently route through the LiteLLM proxy, which
    strips ``cache_control`` (eliminating prompt-cache discounts of 80%+ on
    warm calls) and zeroes ``response_cost`` (breaking cost telemetry).
    Local model prefixes (ollama/, hosted_vllm/, openai/localhost) are
    intentionally exempt.
    """
    if model in _PREFIX_WARNED:
        return
    if any(model.startswith(p) for p in _LOCAL_MODEL_PREFIXES):
        return
    if model.startswith("openrouter/"):
        return
    if not os.getenv("OPENROUTER_API_KEY_IMAS_CODEX"):
        return
    _PREFIX_WARNED.add(model)
    logger.warning(
        "model=%s missing openrouter/ prefix — cache_control will be "
        "stripped and response_cost will be 0. Add the prefix in pyproject "
        "or env override to restore cache discounts and cost telemetry.",
        model,
    )


# ---------------------------------------------------------------------------
# JSON schema format — always convert Pydantic models to json_schema dicts
# ---------------------------------------------------------------------------


def _is_pydantic_model(obj: Any) -> bool:
    """Check if obj is a Pydantic BaseModel class (not instance)."""
    try:
        from pydantic import BaseModel

        return isinstance(obj, type) and issubclass(obj, BaseModel)
    except ImportError:
        return False


def _strip_unsupported_schema_props(schema: dict) -> dict:
    """Recursively strip JSON Schema properties unsupported by some providers.

    Anthropic (via Azure/OpenRouter) rejects several JSON Schema features
    in structured output schemas even with ``strict: false``:

    - ``maxItems``, ``minItems``, ``minLength``, ``maxLength``, ``pattern``,
      ``minimum``, ``maximum`` — constraint keywords
    - ``default`` — default value hints
    - ``title`` — auto-generated by Pydantic, adds noise

    We strip them so the schema works across all providers; our Pydantic
    parsing still validates these constraints.
    """
    unsupported = {
        "maxItems",
        "minItems",
        "minLength",
        "maxLength",
        "pattern",
        "minimum",
        "maximum",
        "exclusiveMinimum",
        "exclusiveMaximum",
        "default",
        "title",
    }
    cleaned: dict = {}
    for key, value in schema.items():
        if key in unsupported:
            continue
        if isinstance(value, dict):
            cleaned[key] = _strip_unsupported_schema_props(value)
        elif isinstance(value, list):
            cleaned[key] = [
                _strip_unsupported_schema_props(item)
                if isinstance(item, dict)
                else item
                for item in value
            ]
        else:
            cleaned[key] = value
    return cleaned


def _to_json_schema_format(model_cls: type) -> dict:
    """Convert a Pydantic model class to a non-strict json_schema format.

    Uses ``strict: false`` so that freeform dicts (``dict[str, str]``)
    are accepted by the API.  Strips provider-unsupported constraints
    (``maxItems``, etc.) — Pydantic parsing validates them instead.
    """
    raw_schema = model_cls.model_json_schema()
    return {
        "type": "json_schema",
        "json_schema": {
            "name": model_cls.__name__,
            "strict": False,
            "schema": _strip_unsupported_schema_props(raw_schema),
        },
    }


# ---------------------------------------------------------------------------
# Tolerant batch-wrapper coercion
# ---------------------------------------------------------------------------


def _wrapper_field(response_model: type) -> tuple[str, type] | None:
    """Return ``(field_name, inner_model)`` if *response_model* is a wrapper.

    A "wrapper" model is a Pydantic model with EXACTLY ONE required field
    whose annotation is ``list[InnerModel]`` for some Pydantic ``InnerModel``.
    Optional additional fields (defaults / ``default_factory``) are allowed —
    the rule is exactly one *required* list-of-model field.

    Returns ``None`` for non-wrappers (single objects, multi-required-field
    models, or list fields whose element type is not a Pydantic model).
    """
    if not _is_pydantic_model(response_model):
        return None
    required = [
        (name, field)
        for name, field in response_model.model_fields.items()
        if field.is_required()
    ]
    if len(required) != 1:
        return None
    name, field = required[0]
    annotation = field.annotation
    if get_origin(annotation) is not list:
        return None
    args = get_args(annotation)
    if len(args) != 1 or not _is_pydantic_model(args[0]):
        return None
    return name, args[0]


def _coerce_to_wrapper(raw: Any, response_model: type) -> BaseModel | None:
    """Coerce a bare inner item/list into a single-list-field wrapper model.

    Some models ignore a batch-wrapper schema and return the bare inner
    object (``{...}``) or a bare list (``[{...}, ...]``) instead of
    ``{"<field>": [...]}``. This wraps such payloads when *response_model*
    is a wrapper (see :func:`_wrapper_field`):

    - ``raw`` is a ``dict`` → try ``{field: [raw]}``
    - ``raw`` is a ``list`` → try ``{field: raw}``

    The candidate is re-validated through the wrapper, which enforces that
    the inner items validate as the element model. Returns the validated
    wrapper instance on success, or ``None`` if *response_model* is not a
    wrapper or the payload cannot be coerced.

    Pure and side-effect free (no logging) so it is trivially unit-testable.
    """
    spec = _wrapper_field(response_model)
    if spec is None:
        return None
    field_name, _inner = spec
    if isinstance(raw, dict):
        candidate = {field_name: [raw]}
    elif isinstance(raw, list):
        candidate = {field_name: raw}
    else:
        return None
    try:
        return response_model.model_validate(candidate)
    except ValidationError:
        return None


def _parse_structured_content(
    content: str, response_model: type, model: str
) -> BaseModel:
    """Validate sanitized LLM *content* against *response_model*.

    On a Pydantic ``ValidationError``, attempt tolerant batch-wrapper
    coercion (a model returning a bare inner item/list when a single-list
    -field wrapper was requested) BEFORE surfacing the error. Successful
    coercion logs a DEBUG line and returns the wrapper instance — no retry
    is consumed.

    If the content is not valid JSON, or coercion does not apply / fails,
    the original ``ValidationError`` is re-raised so the existing retry
    behaviour in the caller is preserved unchanged.
    """
    try:
        return response_model.model_validate_json(content)
    except ValidationError as exc:
        try:
            raw = json.loads(content)
        except ValueError:
            # Not valid JSON at all (e.g. truncated) — coercion can't help.
            raise exc from None
        coerced = _coerce_to_wrapper(raw, response_model)
        if coerced is None:
            raise
        field_name = _wrapper_field(response_model)[0]  # type: ignore[index]
        logger.debug(
            "coerced bare item(s) into %s.%s for model %s",
            response_model.__name__,
            field_name,
            model,
        )
        return coerced


# ---------------------------------------------------------------------------
# Retry / cost helpers
# ---------------------------------------------------------------------------


def _is_retryable(error_msg: str) -> bool:
    """Check if an error message indicates a retryable condition.

    Non-retryable patterns (e.g. vocab-gap validation) take priority over
    retryable patterns so that deterministic failures short-circuit early.
    """
    msg_lower = error_msg.lower()
    if any(pattern in msg_lower for pattern in NON_RETRYABLE_PATTERNS):
        return False
    return any(pattern in msg_lower for pattern in RETRYABLE_PATTERNS)


def _is_local_model(model_id: str) -> bool:
    """Return True if the model is served locally (zero cost)."""
    return any(model_id.startswith(p) for p in _LOCAL_MODEL_PREFIXES)


def extract_cost(response: Any, *, model: str | None = None) -> float:
    """Extract actual LLM cost from a LiteLLM response.

    Priority:
    1. Local models (hosted_vllm/, ollama/) → always 0.0
    2. OpenRouter response_cost from _hidden_params (most accurate)
    3. Fallback: Claude Sonnet rates ($3/$15 per 1M tokens)

    Args:
        response: LiteLLM completion response object
        model: Model identifier used in the request.  When provided,
            local models short-circuit to zero cost.

    Returns:
        Cost in USD.
    """
    # Local/self-hosted models are free at point of use.
    resp_model = model or getattr(response, "model", "") or ""
    if _is_local_model(resp_model):
        return 0.0

    if hasattr(response, "_hidden_params"):
        cost = response._hidden_params.get("response_cost")
        if cost is not None:
            return float(cost)

    usage = getattr(response, "usage", None)
    if usage:
        input_tokens = usage.prompt_tokens or 0
        output_tokens = usage.completion_tokens or 0
        return (input_tokens * 3 + output_tokens * 15) / 1_000_000
    return 0.0


def _extract_cache_fields(ptd: Any) -> tuple[int, int]:
    """Extract cache read/write token counts from prompt_tokens_details.

    Different providers use different field names:
    - litellm formal: ``cache_creation_tokens`` (None by default)
    - OpenRouter extra: ``cache_write_tokens``
    - Both use: ``cached_tokens`` for reads

    Returns ``(cached_read, cache_write)`` with 0 defaults.
    """
    if ptd is None:
        return 0, 0
    cached = getattr(ptd, "cached_tokens", 0) or 0
    # Check both litellm formal field and OpenRouter's extra field
    cache_write = getattr(ptd, "cache_creation_tokens", 0) or 0
    if cache_write == 0:
        cache_write = getattr(ptd, "cache_write_tokens", 0) or 0
    return cached, cache_write


def _log_cache_metrics(response: Any, model: str) -> None:
    """Log prompt cache hit/miss metrics from LLM response usage.

    Providers report cached token counts in ``usage.prompt_tokens_details``.
    This logs at DEBUG level for post-hoc analysis of cache effectiveness
    via the auto-rotating CLI log files.
    """
    usage = getattr(response, "usage", None)
    if not usage:
        return
    ptd = getattr(usage, "prompt_tokens_details", None)
    cached, cache_write = _extract_cache_fields(ptd)
    prompt = getattr(usage, "prompt_tokens", 0) or 0
    completion = getattr(usage, "completion_tokens", 0) or 0
    cost = extract_cost(response, model=model)

    if cached > 0:
        pct = cached / prompt * 100 if prompt > 0 else 0
        logger.debug(
            "LLM cache HIT: %d/%d prompt tokens cached (%.0f%%), "
            "completion=%d, cost=$%.4f, model=%s",
            cached,
            prompt,
            pct,
            completion,
            cost,
            model,
        )
    elif cache_write > 0:
        logger.debug(
            "LLM cache WRITE: %d tokens written, prompt=%d, "
            "completion=%d, cost=$%.4f, model=%s",
            cache_write,
            prompt,
            completion,
            cost,
            model,
        )
    else:
        logger.debug(
            "LLM cache MISS: prompt=%d, completion=%d, cost=$%.4f, model=%s",
            prompt,
            completion,
            cost,
            model,
        )


def extract_cache_tokens(response: Any) -> tuple[int, int]:
    """Extract prompt-cache token counts from an LLM response.

    Providers (OpenRouter, Anthropic, OpenAI) report cached token counts
    in ``usage.prompt_tokens_details``.  This helper mirrors the extraction
    logic of :func:`_log_cache_metrics` but returns the values instead of
    logging them, so callers can accumulate cache statistics.

    Args:
        response: Raw litellm response object.

    Returns:
        ``(cache_read_tokens, cache_creation_tokens)`` — both default to 0
        when the provider does not report caching information.
    """
    usage = getattr(response, "usage", None)
    if not usage:
        return 0, 0
    ptd = getattr(usage, "prompt_tokens_details", None)
    return _extract_cache_fields(ptd)


def _sanitize_content(content: str) -> str:
    """Sanitize LLM response content for JSON parsing.

    Removes control characters, strips markdown code fences, extracts JSON
    from prose wrappers, and fixes surrogate encoding issues.

    Args:
        content: Raw LLM response content string.

    Returns:
        Cleaned string safe for JSON/Pydantic parsing.
    """
    # Strip markdown code fences (```json ... ``` or ``` ... ```)
    content = re.sub(r"^```(?:json)?\s*\n?", "", content.strip())
    content = re.sub(r"\n?```\s*$", "", content)

    # Extract JSON from prose wrappers — LLMs sometimes "think aloud" before
    # or after the JSON object (e.g. "Looking at these paths...\n{...}")
    stripped = content.strip()
    if stripped and stripped[0] not in ("{", "["):
        json_start = _find_json_start(stripped)
        if json_start >= 0:
            content = _extract_balanced_json(stripped, json_start)

    # Remove control characters
    content = re.sub(r"[\x00-\x08\x0b\x0c\x0e-\x1f]", "", content)

    # Fix surrogate encoding issues
    content = content.encode("utf-8", errors="surrogateescape").decode(
        "utf-8", errors="replace"
    )
    return content


def _find_json_start(text: str) -> int:
    """Find the start of a JSON object or array in text.

    Looks for ``{`` or ``[`` that is likely the start of a JSON value,
    skipping occurrences inside obvious prose (e.g. ``{variable}``
    template markers).
    """
    for i, ch in enumerate(text):
        if ch == "{":
            # Skip Jinja/template-like markers: single word in braces
            # e.g. {variable} but not {"key": ...}
            rest = text[i + 1 : i + 50]
            if rest.lstrip().startswith('"') or rest.lstrip().startswith("'"):
                return i
            # Also accept if it looks like the start of JSON with newlines
            if "\n" in text[i : i + 200]:
                return i
            # Fallback: accept any { that's followed by another { or "
            for ch2 in rest:
                if ch2 in ('"', "'", "{", "["):
                    return i
                if ch2 in (" ", "\t", "\n", "\r"):
                    continue
                break
        elif ch == "[":
            return i
    return -1


def _extract_balanced_json(text: str, start: int) -> str:
    """Extract a balanced JSON object/array starting at *start*.

    Uses a simple brace/bracket counter that respects JSON strings
    (skipping escaped characters inside double quotes).
    Falls back to ``text[start:]`` if no balanced close is found.
    """
    open_ch = text[start]
    close_ch = "}" if open_ch == "{" else "]"
    depth = 0
    in_string = False
    i = start
    while i < len(text):
        ch = text[i]
        if in_string:
            if ch == "\\" and i + 1 < len(text):
                i += 2  # skip escaped char
                continue
            if ch == '"':
                in_string = False
        else:
            if ch == '"':
                in_string = True
            elif ch == open_ch:
                depth += 1
            elif ch == close_ch:
                depth -= 1
                if depth == 0:
                    return text[start : i + 1]
        i += 1
    # No balanced close found — return from start to end
    return text[start:]


# ---------------------------------------------------------------------------
# Prompt caching (data-driven from config/prompt_caching.yaml)
# ---------------------------------------------------------------------------

_PROMPT_CACHING_CONFIG = (
    Path(__file__).resolve().parents[2] / "config" / "prompt_caching.yaml"
)


@lru_cache(maxsize=1)
def _cache_control_patterns() -> tuple[str, ...]:
    """Match patterns for providers needing explicit cache_control breakpoints."""
    with _PROMPT_CACHING_CONFIG.open() as f:
        config = yaml.safe_load(f)
    patterns: list[str] = []
    for provider in config.get("providers", {}).values():
        patterns.extend(provider.get("match", []))
    return tuple(patterns)


def _supports_cache_control(model: str) -> bool:
    """Check if a model needs explicit cache_control breakpoints."""
    model_lower = model.lower()
    return any(p in model_lower for p in _cache_control_patterns())


def inject_cache_control(
    messages: list[dict[str, Any]],
) -> list[dict[str, Any]]:
    """Add cache_control breakpoints to system messages for prompt caching.

    Converts the last system message's content to a content-block list
    with ``cache_control: {"type": "ephemeral"}`` on the last block.
    Provider support is configured in ``config/prompt_caching.yaml``.

    The default 5-minute ``ephemeral`` TTL is optimal for discovery
    workers that fire calls every 1-3 seconds, keeping the cache warm
    continuously throughout a CLI run.

    Args:
        messages: Chat messages (not mutated — a shallow copy is returned).

    Returns:
        New message list with cache_control injected on the last system message.
    """
    messages = [m.copy() for m in messages]

    # Walk backwards to find the last system message
    for i in range(len(messages) - 1, -1, -1):
        if messages[i].get("role") == "system":
            content = messages[i]["content"]
            if isinstance(content, str):
                messages[i]["content"] = [
                    {
                        "type": "text",
                        "text": content,
                        "cache_control": {"type": "ephemeral"},
                    }
                ]
            elif isinstance(content, list) and content:
                # Already in block format — add cache_control to last block
                content = [block.copy() for block in content]
                content[-1]["cache_control"] = {"type": "ephemeral"}
                messages[i]["content"] = content
            break

    return messages


# Instruction appended to satisfy providers that require the literal word
# "json" in the prompt before honouring a response_format / json_schema
# request (Alibaba Dashscope, which backs qwen, rejects the call otherwise:
# "'messages' must contain the word 'json' in some form, to use
# 'response_format'"). Harmless for every other provider — they already
# receive the schema and ignore the extra sentence.
_JSON_INSTRUCTION = "Respond with a single valid JSON object."


def _message_text(content: Any) -> str:
    """Flatten a message ``content`` field to plain text for substring scans.

    Handles both the plain-string form and the list-of-content-block form
    (each block a dict carrying a ``text`` key, optionally with
    ``cache_control``). Non-text blocks contribute nothing.
    """
    if isinstance(content, str):
        return content
    if isinstance(content, list):
        return " ".join(
            str(block.get("text", "")) for block in content if isinstance(block, dict)
        )
    return ""


def _messages_mention_json(messages: list[dict[str, Any]]) -> bool:
    """Return True if any message's text already contains "json" (case-insensitive)."""
    return any("json" in _message_text(m.get("content")).lower() for m in messages)


def _ensure_json_in_messages(
    messages: list[dict[str, Any]],
) -> list[dict[str, Any]]:
    """Ensure the word "json" appears in *messages* for response_format calls.

    Some providers (Dashscope/qwen) hard-reject a ``response_format`` /
    json_schema request unless the literal word "json" is present in the
    prompt. This appends a minimal JSON instruction so the guard passes,
    while remaining:

    - **Idempotent** — a no-op when "json" is already present anywhere in
      the system/user text (many prompts already mention JSON).
    - **Cache-safe** — appends to the existing text of the last system
      message rather than adding a new uncached message, preserving any
      ``cache_control`` breakpoint prefix. The append targets the LAST text
      block (where ``inject_cache_control`` places the breakpoint), keeping
      the cacheable prefix byte-identical.
    - **Provider-agnostic** — every other provider already gets the schema
      and treats the extra sentence as harmless guidance.

    When no system message exists, one is prepended carrying the
    instruction. The input list is not mutated — a shallow copy is returned.
    """
    if not messages or _messages_mention_json(messages):
        return messages

    messages = [m.copy() for m in messages]

    # Append to the last system message to preserve any cache_control prefix.
    for i in range(len(messages) - 1, -1, -1):
        if messages[i].get("role") != "system":
            continue
        content = messages[i]["content"]
        if isinstance(content, str):
            sep = "" if content.endswith(("\n", " ")) or not content else " "
            messages[i]["content"] = f"{content}{sep}{_JSON_INSTRUCTION}"
        elif isinstance(content, list) and content:
            # Append to the last text block, keeping its cache_control intact.
            content = [block.copy() for block in content]
            for j in range(len(content) - 1, -1, -1):
                block = content[j]
                if isinstance(block, dict) and "text" in block:
                    text = str(block["text"])
                    sep = "" if text.endswith(("\n", " ")) or not text else " "
                    block["text"] = f"{text}{sep}{_JSON_INSTRUCTION}"
                    break
            else:
                # No text block to append to — add a fresh text block.
                content.append({"type": "text", "text": _JSON_INSTRUCTION})
            messages[i]["content"] = content
        else:
            # Empty/None content — set the instruction directly.
            messages[i]["content"] = _JSON_INSTRUCTION
        return messages

    # No system message — prepend one carrying the instruction.
    return [{"role": "system", "content": _JSON_INSTRUCTION}, *messages]


def _build_kwargs(
    model: str,
    api_key: str,
    messages: list[dict[str, Any]],
    response_format: type[BaseModel] | None,
    max_tokens: int | None,
    temperature: float | None,
    timeout: int | None,
    *,
    service: str = "untagged",
    api_base: str | None = None,
    api_key_override: str | None = None,
    reasoning_effort: str | None = None,
) -> dict[str, Any]:
    """Build litellm completion kwargs with model-aware defaults.

    When *api_base* is provided (from ``get_model_config()``), the call
    is routed directly to that endpoint — bypassing both the LiteLLM
    proxy and OpenRouter.  This enables local/self-hosted models.

    When max_tokens or timeout are not explicitly set, uses
    model-family defaults from MODEL_TOKEN_LIMITS.
    """
    from imas_codex.settings import (
        get_llm_location,
        get_llm_proxy_url,
        get_model_endpoint,
    )

    _warn_if_missing_openrouter_prefix(model)

    limits = get_model_limits(model)

    # Auto-resolve endpoint from the model registry if not explicitly passed
    if not api_base:
        endpoint = get_model_endpoint(model)
        if endpoint:
            api_base = endpoint["api_base"]
            if not api_key_override and endpoint.get("api_key_env"):
                api_key_override = os.getenv(endpoint["api_key_env"])

    # ── Per-section endpoint override (local/self-hosted models) ──────
    if api_base:
        # Direct call to a custom endpoint — no proxy, no OpenRouter.
        effective_key = api_key_override or api_key
        kwargs: dict[str, Any] = {
            "model": model,
            "api_key": effective_key,
            "api_base": api_base,
            "max_tokens": max_tokens
            if max_tokens is not None
            else limits["max_tokens"],
            "timeout": timeout if timeout is not None else limits["timeout"],
            "messages": messages,
        }
        logger.debug(
            "Direct endpoint for %s → %s (proxy/OpenRouter bypassed)",
            model,
            api_base,
        )
    else:
        # ── Standard routing: proxy vs OpenRouter direct ─────────────
        llm_location = get_llm_location()

        # Inject cache_control for models that support explicit breakpoints.
        supports_cache = _supports_cache_control(model)
        if supports_cache:
            messages = inject_cache_control(messages)

        use_proxy = llm_location != "local" or bool(os.getenv("LITELLM_PROXY_URL"))
        has_direct_key = bool(os.getenv("OPENROUTER_API_KEY_IMAS_CODEX"))
        is_openrouter_model = "openrouter/" in model.lower() or any(
            p in model.lower()
            for p in (
                "anthropic/",
                "google/",
                "openai/",
                "deepseek/",
                "meta-llama/",
                "moonshotai/",
                "qwen/",
                "mistralai/",
            )
        )
        bypass_proxy = has_direct_key and is_openrouter_model

        if use_proxy and not bypass_proxy:
            proxy_url = get_llm_proxy_url()
            model_id = f"openai/{ensure_model_prefix(model)}"
            proxy_key = os.getenv("LITELLM_API_KEY") or os.getenv(
                "LITELLM_MASTER_KEY", api_key
            )
            kwargs = {
                "model": model_id,
                "api_key": proxy_key,
                "api_base": proxy_url,
                "max_tokens": max_tokens
                if max_tokens is not None
                else limits["max_tokens"],
                "timeout": timeout if timeout is not None else limits["timeout"],
                "messages": messages,
            }
        else:
            model_id = ensure_model_prefix(model)
            direct_key = get_api_key_for_service(service) if bypass_proxy else api_key

            kwargs = {
                "model": model_id,
                "api_key": direct_key,
                "max_tokens": max_tokens
                if max_tokens is not None
                else limits["max_tokens"],
                "timeout": timeout if timeout is not None else limits["timeout"],
                "messages": messages,
            }

            if bypass_proxy:
                logger.debug(
                    "Bypassing proxy for %s (cache_control preserved)", model_id
                )

    if response_format is not None:
        # Always convert Pydantic models to explicit json_schema dicts.
        # Passing raw Pydantic classes through LiteLLM proxy → OpenRouter
        # does not reliably enforce structured output for all providers.
        # The explicit schema dict is provider-agnostic; strict=false lets
        # the model use it as guidance while our Pydantic parsing validates.
        if _is_pydantic_model(response_format):
            kwargs["response_format"] = _to_json_schema_format(response_format)
        else:
            kwargs["response_format"] = response_format
        # Dashscope (qwen) rejects response_format unless the prompt mentions
        # "json". Inject a minimal instruction on the FINAL messages — covers
        # every branch (api_base / proxy / direct) and preserves cache_control.
        kwargs["messages"] = _ensure_json_in_messages(kwargs["messages"])
    if temperature is not None:
        # GPT-5.x models reject temperature=0.0 with a 400 error; clamp to None
        # (provider default) so callers that pin temperature=0.0 don't break.
        if temperature == 0.0 and re.search(r"gpt-5", model):
            logger.debug("Clamping temperature=0.0 → None for GPT-5.x model %s", model)
        else:
            kwargs["temperature"] = temperature

    # Per-service X-Title for OpenRouter dashboard visibility.
    # Client extra_headers shallow-replaces proxy config extra_headers,
    # so per-service titles override the proxy fallback "imas-codex".
    title = f"imas-codex:{service}"
    kwargs["extra_headers"] = {
        "X-Title": title,
        "HTTP-Referer": "https://github.com/iterorganization/imas-codex",
    }

    kwargs["metadata"] = {
        "service": service,
    }

    # Reasoning-effort passthrough — provider-shaped:
    #
    # * ``hosted_vllm/`` (local vLLM, e.g. DeepSeek V4 served by ambix):
    #   vLLM's OpenAI server takes ``chat_template_kwargs`` in the request
    #   body. DeepSeek V4 thinking mode requires
    #   ``{"thinking": true, "reasoning_effort": "high"|"max"}`` — WITHOUT
    #   this the model silently runs in non-think mode, which materially
    #   degrades generation quality.
    # * Everything else: OpenRouter's NATIVE unified field
    #   (``reasoning: {effort: low|medium|high}``) carried in ``extra_body``,
    #   which litellm forwards verbatim. We deliberately avoid litellm's
    #   ``reasoning_effort`` kwarg — its per-provider mapping raises
    #   UnsupportedParamsError for many OpenRouter models. OpenRouter ignores
    #   the field for models without reasoning. None = provider default.
    if reasoning_effort is not None:
        extra_body = kwargs.setdefault("extra_body", {})
        if model.startswith("hosted_vllm/"):
            extra_body["chat_template_kwargs"] = {
                "thinking": True,
                "reasoning_effort": reasoning_effort,
            }
        else:
            extra_body["reasoning"] = {"effort": reasoning_effort}

    return kwargs


# ---------------------------------------------------------------------------
# Structured output: call + parse in shared retry loop
# ---------------------------------------------------------------------------


def call_llm_structured(
    model: str,
    messages: list[dict[str, Any]],
    response_model: type[BaseModel],
    *,
    max_tokens: int | None = None,
    temperature: float | None = None,
    timeout: int | None = None,
    max_retries: int = DEFAULT_MAX_RETRIES,
    retry_base_delay: float = DEFAULT_RETRY_BASE_DELAY,
    service: str = "untagged",
    reasoning_effort: str | None = None,
) -> LLMResult:
    """Call LLM and parse structured output, retrying on both API and parse errors.

    Wraps the LLM call and Pydantic parsing in a single retry loop so that
    truncated JSON or validation errors trigger a fresh attempt. Cost is
    accumulated across retries since API calls are billed regardless.

    This pattern was proven in the wiki scoring pipeline and is shared
    across all discovery domains.

    Args:
        model: Model identifier (e.g., "google/gemini-3-flash-preview").
        messages: Chat messages [{"role": ..., "content": ...}].
        response_model: Pydantic model for structured output parsing.
        max_tokens: Max output tokens (None = model-family default).
        temperature: Sampling temperature (None = model default).
        timeout: Request timeout seconds (None = model-family default).
        max_retries: Maximum retry attempts.
        retry_base_delay: Base delay for exponential backoff (seconds).

    Returns:
        :class:`LLMResult` — backward-compatible with 3-tuple unpacking
        ``(parsed_model, total_cost_usd, total_tokens)`` and also carries
        ``cache_read_tokens`` / ``cache_creation_tokens``.

    Raises:
        ValueError: If response parsing fails after all retries.
        Exception: Non-retryable errors from LLM provider.
    """
    import litellm

    suppress_litellm_noise()

    api_key = get_api_key()
    kwargs = _build_kwargs(
        model,
        api_key,
        messages,
        response_model,
        max_tokens,
        temperature,
        timeout,
        service=service,
        reasoning_effort=reasoning_effort,
    )

    last_error: Exception | None = None
    total_cost = 0.0

    for attempt in range(max_retries):
        try:
            response = litellm.completion(**kwargs)
            total_cost += extract_cost(response, model=model)
            _log_cache_metrics(response, model)

            # Parse response content through Pydantic
            content = response.choices[0].message.content
            if not content:
                raise ValueError("LLM returned empty response content")

            content = _sanitize_content(content)
            parsed = _parse_structured_content(content, response_model, model)

            total_tokens = (
                response.usage.prompt_tokens + response.usage.completion_tokens
            )
            input_tokens = response.usage.prompt_tokens or 0
            output_tokens = response.usage.completion_tokens or 0
            cache_read, cache_creation = extract_cache_tokens(response)
            return LLMResult(
                parsed,
                total_cost,
                total_tokens,
                cache_read,
                cache_creation,
                input_tokens=input_tokens,
                output_tokens=output_tokens,
            )

        except Exception as e:
            last_error = e
            error_msg = str(e)
            if _is_budget_exhausted(error_msg):
                raise ProviderBudgetExhausted(
                    f"LLM provider budget exhausted: {error_msg[:200]}"
                ) from e
            if _is_retryable(error_msg) and attempt < max_retries - 1:
                delay = retry_base_delay * (2**attempt)
                logger.debug(
                    "LLM error (attempt %d/%d): %s. Retrying in %.1fs...",
                    attempt + 1,
                    max_retries,
                    error_msg[:100],
                    delay,
                )
                time.sleep(delay)
            elif attempt == max_retries - 1:
                logger.error(
                    "LLM failed after %d attempts: %s",
                    max_retries,
                    error_msg[:200],
                )
                raise ValueError(
                    f"LLM failed after {max_retries} attempts: {error_msg[:200]}"
                ) from e
            else:
                raise

    raise last_error  # type: ignore[misc]


async def acall_llm_structured(
    model: str,
    messages: list[dict[str, Any]],
    response_model: type[BaseModel],
    *,
    max_tokens: int | None = None,
    temperature: float | None = None,
    timeout: int | None = None,
    max_retries: int = DEFAULT_MAX_RETRIES,
    retry_base_delay: float = DEFAULT_RETRY_BASE_DELAY,
    service: str = "untagged",
    reasoning_effort: str | None = None,
) -> LLMResult:
    """Async version of call_llm_structured.

    Identical retry+parse semantics using litellm.acompletion() and
    asyncio.sleep() for non-blocking backoff.

    Args:
        model: Model identifier (e.g., "google/gemini-3-flash-preview").
        messages: Chat messages [{"role": ..., "content": ...}].
        response_model: Pydantic model for structured output parsing.
        max_tokens: Max output tokens (None = model-family default).
        temperature: Sampling temperature (None = model default).
        timeout: Request timeout seconds (None = model-family default).
        max_retries: Maximum retry attempts.
        retry_base_delay: Base delay for exponential backoff (seconds).

    Returns:
        :class:`LLMResult` — backward-compatible with 3-tuple unpacking
        ``(parsed_model, total_cost_usd, total_tokens)`` and also carries
        ``cache_read_tokens`` / ``cache_creation_tokens``.

    Raises:
        ValueError: If response parsing fails after all retries.
        Exception: Non-retryable errors from LLM provider.
    """
    import litellm

    suppress_litellm_noise()

    api_key = get_api_key()
    kwargs = _build_kwargs(
        model,
        api_key,
        messages,
        response_model,
        max_tokens,
        temperature,
        timeout,
        service=service,
        reasoning_effort=reasoning_effort,
    )

    last_error: Exception | None = None
    total_cost = 0.0

    for attempt in range(max_retries):
        try:
            response = await litellm.acompletion(**kwargs)
            total_cost += extract_cost(response, model=model)
            _log_cache_metrics(response, model)

            # Parse response content through Pydantic
            content = response.choices[0].message.content
            if not content:
                raise ValueError("LLM returned empty response content")

            content = _sanitize_content(content)
            parsed = _parse_structured_content(content, response_model, model)

            total_tokens = (
                response.usage.prompt_tokens + response.usage.completion_tokens
            )
            input_tokens = response.usage.prompt_tokens or 0
            output_tokens = response.usage.completion_tokens or 0
            cache_read, cache_creation = extract_cache_tokens(response)
            return LLMResult(
                parsed,
                total_cost,
                total_tokens,
                cache_read,
                cache_creation,
                input_tokens=input_tokens,
                output_tokens=output_tokens,
            )

        except Exception as e:
            last_error = e
            error_msg = str(e)
            if _is_budget_exhausted(error_msg):
                raise ProviderBudgetExhausted(
                    f"LLM provider budget exhausted: {error_msg[:200]}"
                ) from e
            if _is_retryable(error_msg) and attempt < max_retries - 1:
                delay = retry_base_delay * (2**attempt)
                logger.debug(
                    "LLM error (attempt %d/%d): %s. Retrying in %.1fs...",
                    attempt + 1,
                    max_retries,
                    error_msg[:100],
                    delay,
                )
                await asyncio.sleep(delay)
            elif attempt == max_retries - 1:
                logger.error(
                    "LLM failed after %d attempts: %s",
                    max_retries,
                    error_msg[:200],
                )
                raise ValueError(
                    f"LLM failed after {max_retries} attempts: {error_msg[:200]}"
                ) from e
            else:
                raise

    raise last_error  # type: ignore[misc]


# ---------------------------------------------------------------------------
# Raw LLM calls (when caller needs custom response handling)
# ---------------------------------------------------------------------------


def call_llm(
    model: str,
    messages: list[dict[str, Any]],
    response_format: type[BaseModel] | None = None,
    *,
    max_tokens: int | None = None,
    temperature: float | None = None,
    timeout: int | None = None,
    max_retries: int = DEFAULT_MAX_RETRIES,
    retry_base_delay: float = DEFAULT_RETRY_BASE_DELAY,
    service: str = "untagged",
) -> tuple[Any, float]:
    """Call LLM synchronously with retry logic and cost tracking.

    Returns the raw LiteLLM response for callers that need custom
    parsing. Prefer call_llm_structured() for Pydantic models.

    Args:
        model: Model identifier (e.g., "google/gemini-3-flash-preview").
        messages: Chat messages [{"role": ..., "content": ...}].
        response_format: Optional Pydantic model for structured output.
        max_tokens: Max output tokens (None = model-family default).
        temperature: Sampling temperature (None = model default).
        timeout: Request timeout seconds (None = model-family default).
        max_retries: Maximum retry attempts.
        retry_base_delay: Base delay for exponential backoff (seconds).

    Returns:
        Tuple of (litellm_response, cost_usd).

    Raises:
        ValueError: If API key is not set.
        Exception: Non-retryable errors from LLM provider.
    """
    import litellm

    suppress_litellm_noise()

    api_key = get_api_key()
    kwargs = _build_kwargs(
        model,
        api_key,
        messages,
        response_format,
        max_tokens,
        temperature,
        timeout,
        service=service,
    )

    last_error: Exception | None = None
    for attempt in range(max_retries):
        try:
            response = litellm.completion(**kwargs)
            cost = extract_cost(response, model=model)
            _log_cache_metrics(response, model)
            return response, cost
        except Exception as e:
            last_error = e
            error_msg = str(e)
            if _is_budget_exhausted(error_msg):
                raise ProviderBudgetExhausted(
                    f"LLM provider budget exhausted: {error_msg[:200]}"
                ) from e
            if _is_retryable(error_msg) and attempt < max_retries - 1:
                delay = retry_base_delay * (2**attempt)
                logger.debug(
                    "LLM rate limited (attempt %d/%d): %s. Retrying in %.1fs...",
                    attempt + 1,
                    max_retries,
                    error_msg[:100],
                    delay,
                )
                time.sleep(delay)
            elif attempt == max_retries - 1:
                logger.error(
                    "LLM failed after %d attempts: %s",
                    max_retries,
                    error_msg[:200],
                )
                raise
            else:
                raise

    raise last_error  # type: ignore[misc]


async def acall_llm(
    model: str,
    messages: list[dict[str, Any]],
    response_format: type[BaseModel] | None = None,
    *,
    max_tokens: int | None = None,
    temperature: float | None = None,
    timeout: int | None = None,
    max_retries: int = DEFAULT_MAX_RETRIES,
    retry_base_delay: float = DEFAULT_RETRY_BASE_DELAY,
    service: str = "untagged",
) -> tuple[Any, float]:
    """Call LLM asynchronously with retry logic and cost tracking.

    Returns the raw LiteLLM response for callers that need custom
    parsing. Prefer acall_llm_structured() for Pydantic models.

    Args:
        model: Model identifier (e.g., "google/gemini-3-flash-preview").
        messages: Chat messages [{"role": ..., "content": ...}].
        response_format: Optional Pydantic model for structured output.
        max_tokens: Max output tokens (None = model-family default).
        temperature: Sampling temperature (None = model default).
        timeout: Request timeout seconds (None = model-family default).
        max_retries: Maximum retry attempts.
        retry_base_delay: Base delay for exponential backoff (seconds).

    Returns:
        Tuple of (litellm_response, cost_usd).

    Raises:
        ValueError: If API key is not set.
        Exception: Non-retryable errors from LLM provider.
    """
    import litellm

    suppress_litellm_noise()

    api_key = get_api_key()
    kwargs = _build_kwargs(
        model,
        api_key,
        messages,
        response_format,
        max_tokens,
        temperature,
        timeout,
        service=service,
    )

    last_error: Exception | None = None
    for attempt in range(max_retries):
        try:
            response = await litellm.acompletion(**kwargs)
            cost = extract_cost(response, model=model)
            _log_cache_metrics(response, model)
            return response, cost
        except Exception as e:
            last_error = e
            error_msg = str(e)
            if _is_budget_exhausted(error_msg):
                raise ProviderBudgetExhausted(
                    f"LLM provider budget exhausted: {error_msg[:200]}"
                ) from e
            if _is_retryable(error_msg) and attempt < max_retries - 1:
                delay = retry_base_delay * (2**attempt)
                logger.debug(
                    "LLM rate limited (attempt %d/%d): %s. Retrying in %.1fs...",
                    attempt + 1,
                    max_retries,
                    error_msg[:100],
                    delay,
                )
                await asyncio.sleep(delay)
            elif attempt == max_retries - 1:
                logger.error(
                    "LLM failed after %d attempts: %s",
                    max_retries,
                    error_msg[:200],
                )
                raise ValueError(
                    f"LLM failed after {max_retries} attempts: {error_msg[:200]}"
                ) from e
            else:
                raise

    raise last_error  # type: ignore[misc]
