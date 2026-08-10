"""Immutable structured-provider request construction for typed dispatch."""

from __future__ import annotations

import hashlib
from base64 import b64decode
from binascii import Error as Base64Error
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


def _redact(value: Any) -> Any:
    if isinstance(value, Mapping):
        redacted: dict[str, Any] = {}
        for key, item in value.items():
            name = str(key)
            if name.lower() in _SECRET_KEYS:
                redacted[name] = "<redacted>"
            elif name == "url" and isinstance(item, str) and item.startswith("data:"):
                header, separator, encoded = item.partition(",")
                if not separator or ";base64" not in header:
                    raise ValueError("Typed attachment data URL must use base64")
                try:
                    content = b64decode(encoded, validate=True)
                except (Base64Error, ValueError) as exc:
                    raise ValueError("Typed attachment data URL is invalid") from exc
                media_type = header.removeprefix("data:").removesuffix(";base64")
                digest = hashlib.sha256(content).hexdigest()
                redacted[name] = (
                    f"data:{media_type};sha256={digest};bytes={len(content)}"
                )
            else:
                redacted[name] = _redact(item)
        return redacted
    if isinstance(value, tuple | list):
        return [_redact(item) for item in value]
    return value


@dataclass(frozen=True, slots=True)
class FrozenWireRequest:
    """The exact post-routing request sent on every typed transport attempt."""

    transport_kwargs: Mapping[str, Any]
    redacted_payload: Mapping[str, Any]
    request_digest: str
    response_model_identity: str
    response_schema_digest: str

    def transport_copy(self) -> dict[str, Any]:
        """Return the exact mutable shape required by the provider client."""
        return _thaw(self.transport_kwargs)


def response_model_identity(response_model: type[BaseModel]) -> str:
    """Return a collision-resistant fully-qualified Pydantic model identity."""
    return f"{response_model.__module__}:{response_model.__qualname__}"


def response_schema_digest(response_model: type[BaseModel]) -> str:
    """Digest the exact canonical response JSON schema."""
    return canonical_fingerprint(response_model.model_json_schema())


def build_frozen_wire_request(
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
) -> FrozenWireRequest:
    """Apply every transport transformation once and freeze the resulting object."""
    from imas_codex.discovery.base.llm import _build_kwargs, get_api_key

    kwargs = _build_kwargs(
        model,
        get_api_key(),
        [_thaw(message) for message in messages],
        response_model,
        max_output_tokens,
        temperature,
        timeout,
        service=service,
        reasoning_effort=reasoning_effort,
        typed_max_price=dict(provider_max_price) if provider_max_price else None,
    )
    identity = response_model_identity(response_model)
    schema_digest = response_schema_digest(response_model)
    frozen = _freeze(kwargs)
    redacted = _freeze(_redact(frozen))
    digest = hashlib.sha256(canonical_json(redacted).encode("utf-8")).hexdigest()
    return FrozenWireRequest(
        transport_kwargs=frozen,
        redacted_payload=redacted,
        request_digest=digest,
        response_model_identity=identity,
        response_schema_digest=schema_digest,
    )


__all__ = [
    "FrozenWireRequest",
    "build_frozen_wire_request",
    "response_model_identity",
    "response_schema_digest",
]
