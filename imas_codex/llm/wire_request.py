"""Immutable, redacted structured-provider request identities."""

from __future__ import annotations

import hashlib
import os
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from types import MappingProxyType
from typing import Any

from pydantic import BaseModel

from imas_codex.llm.context_envelope import canonical_fingerprint, canonical_json

_SECRET_KEYS = frozenset({"api_key", "authorization"})


def _freeze(value: Any) -> Any:
    if isinstance(value, Mapping):
        return MappingProxyType(
            {str(key): _freeze(item) for key, item in value.items()}
        )
    if isinstance(value, Sequence) and not isinstance(value, str | bytes | bytearray):
        return tuple(_freeze(item) for item in value)
    return value


def _thaw(value: Any) -> Any:
    if isinstance(value, Mapping):
        return {str(key): _thaw(item) for key, item in value.items()}
    if isinstance(value, tuple):
        return [_thaw(item) for item in value]
    return value


def _redact(
    value: Any,
    attachment_redactions: Mapping[str, str],
) -> Any:
    if isinstance(value, Mapping):
        redacted: dict[str, Any] = {}
        for key, item in value.items():
            name = str(key)
            if name.lower() in _SECRET_KEYS:
                redacted[name] = "<redacted>"
            elif name == "url" and isinstance(item, str) and item.startswith("data:"):
                replacement = attachment_redactions.get(item)
                if replacement is None:
                    raise ValueError(
                        "Typed attachment data URL lacks a validated redaction identity"
                    )
                redacted[name] = replacement
            else:
                redacted[name] = _redact(item, attachment_redactions)
        return redacted
    if isinstance(value, tuple | list):
        return [_redact(item, attachment_redactions) for item in value]
    return value


def _without_secrets(value: Any) -> Any:
    if isinstance(value, Mapping):
        return {
            str(key): _without_secrets(item)
            for key, item in value.items()
            if str(key).lower() not in _SECRET_KEYS
        }
    if isinstance(value, tuple | list):
        return [_without_secrets(item) for item in value]
    return value


@dataclass(frozen=True, slots=True)
class FrozenWireRequest:
    """Public redacted identity of one exact private transport request."""

    redacted_payload: Mapping[str, Any]
    request_digest: str
    response_model_identity: str
    response_schema_digest: str
    credential_source_identity: str
    endpoint_contract: str


class _FrozenTransportHandle:
    """Module-private opaque access to credential-bearing transport state."""

    __slots__ = ("__transport_kwargs", "__tokenization_payload")

    def __init__(self, transport_kwargs: Any, tokenization_payload: Any) -> None:
        self.__transport_kwargs = transport_kwargs
        self.__tokenization_payload = tokenization_payload

    def _transport_copy(self) -> dict[str, Any]:
        return _thaw(self.__transport_kwargs)

    def _tokenization_copy(self) -> dict[str, Any]:
        return _thaw(self.__tokenization_payload)


def response_model_identity(response_model: type[BaseModel]) -> str:
    """Return a collision-resistant fully-qualified Pydantic model identity."""
    return f"{response_model.__module__}:{response_model.__qualname__}"


def response_schema_digest(response_model: type[BaseModel]) -> str:
    """Digest the exact canonical response JSON schema."""
    return canonical_fingerprint(response_model.model_json_schema())


def _credential_identity(source_name: str, endpoint: str, api_key: str) -> str:
    key_id = hashlib.sha256(
        f"{source_name}\0{endpoint}\0{api_key}".encode()
    ).hexdigest()
    return f"{source_name}:sha256:{key_id}"


def _build_frozen_wire_request(
    *,
    model: str,
    messages: Sequence[Mapping[str, Any]],
    response_model: type[BaseModel],
    max_output_tokens: int,
    temperature: float | None,
    timeout: int | None,
    service: str,
    reasoning_effort: str | None,
    provider_max_price: Mapping[str, float] | None,
    provider_selector: str | None,
    configured_model: str,
    zero_cost_local: bool,
    api_base: str | None,
    api_key_env: str | None,
    endpoint_class: str | None,
    attachment_redactions: Mapping[str, str],
) -> tuple[FrozenWireRequest, _FrozenTransportHandle]:
    """Apply transport transformations once and return public/private halves."""
    from imas_codex.discovery.base.llm import (
        _build_kwargs,
        get_api_key_for_service_with_source,
    )

    endpoint_contract = "local-free" if zero_cost_local else "direct-openrouter"
    if zero_cost_local:
        if endpoint_class != "local-free" or not api_base or not api_key_env:
            raise ValueError("Local typed route lacks its exact endpoint contract")
        api_key = os.getenv(api_key_env, "")
        if not api_key:
            raise ValueError(
                f"Local typed route credential source {api_key_env!r} is unavailable"
            )
        credential_source = api_key_env
        endpoint_identity = api_base
    else:
        if any((api_base, api_key_env, endpoint_class)):
            raise ValueError("Paid typed route cannot use a custom endpoint")
        if not model.startswith("openrouter/") or not configured_model.startswith(
            "openrouter/"
        ):
            raise ValueError(
                "Paid typed route lacks an exact OpenRouter model identity"
            )
        if not provider_selector:
            raise ValueError("Paid typed route lacks an exact provider selector")
        api_key, credential_source = get_api_key_for_service_with_source(service)
        endpoint_identity = "https://openrouter.ai/api/v1"
    kwargs = _build_kwargs(
        model,
        api_key,
        [_thaw(message) for message in messages],
        response_model,
        max_output_tokens,
        temperature,
        timeout,
        service=service,
        api_base=api_base,
        api_key_override=api_key if zero_cost_local else None,
        reasoning_effort=reasoning_effort,
        typed_max_price=dict(provider_max_price) if provider_max_price else None,
        typed_provider_selector=provider_selector,
        typed_endpoint_contract=endpoint_contract,
        typed_resolved_api_key=api_key,
    )
    identity = response_model_identity(response_model)
    schema_digest = response_schema_digest(response_model)
    credential_identity = _credential_identity(
        credential_source, endpoint_identity, api_key
    )
    frozen = _freeze(kwargs)
    tokenization_payload = _freeze(_without_secrets(frozen))
    redacted = _freeze(_redact(frozen, attachment_redactions))
    digest = hashlib.sha256(
        canonical_json(
            {
                "endpoint_contract": endpoint_contract,
                "configured_model": configured_model,
                "credential_source_identity": credential_identity,
                "payload": redacted,
            }
        ).encode("utf-8")
    ).hexdigest()
    public = FrozenWireRequest(
        redacted_payload=redacted,
        request_digest=digest,
        response_model_identity=identity,
        response_schema_digest=schema_digest,
        credential_source_identity=credential_identity,
        endpoint_contract=endpoint_contract,
    )
    return public, _FrozenTransportHandle(frozen, tokenization_payload)


__all__ = [
    "FrozenWireRequest",
    "response_model_identity",
    "response_schema_digest",
]
