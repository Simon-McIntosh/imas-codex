"""Immutable registry-owned policy specifications for typed LLM dispatch.

Policy resources are split by domain so a migration can add its own JSON file
without editing a shared Python tuple.  The checked-in resource is the trusted
boundary: business callers select a stable callsite and may supply only runtime
axes that its policy explicitly permits.
"""

from __future__ import annotations

import importlib
import json
import math
import re
from collections.abc import Callable, Iterable, Mapping
from dataclasses import dataclass
from pathlib import Path
from types import MappingProxyType
from typing import Any

from pydantic import BaseModel

from imas_codex.llm.callsite_registry import get_callsite_registration

DISPATCH_POLICY_DIR = Path(__file__).parent / "config" / "dispatch_policies"


class DispatchPolicyRegistryError(ValueError):
    """A trusted dispatch-policy resource is absent, invalid, or ambiguous."""


@dataclass(frozen=True, slots=True)
class TemplateRoleSpec:
    """One template in the exact message-role order owned by a policy."""

    role: str
    name: str
    source_version: str


@dataclass(frozen=True, slots=True)
class ClaimChannelSpec:
    """Exact kinds and scopes admitted to one typed context channel."""

    channel: str
    kinds: frozenset[str]
    scopes: frozenset[str]


@dataclass(frozen=True, slots=True)
class StaticProviderSpec:
    """Versioned schema, grammar, or other immutable provider snapshot."""

    name: str
    kind: str
    source_version: str


@dataclass(frozen=True, slots=True)
class AttachmentPolicySpec:
    """Registry limits for multimodal content carried by one request."""

    allowed_media_types: frozenset[str] = frozenset()
    max_count: int = 0
    max_bytes_each: int = 0
    max_bytes_total: int = 0
    max_width: int = 0
    max_height: int = 0


@dataclass(frozen=True, slots=True)
class DispatchPolicySpec:
    """Complete immutable authority, request, and spend policy for a callsite."""

    policy_id: str
    source_version: str
    callsite_id: str
    service: str
    seat: str
    task_kind: str
    templates: tuple[TemplateRoleSpec, ...]
    response_model_path: str
    model_section: str
    tokenizer_path: str
    tokenizer_key: str
    identifier_pattern: str
    channels: tuple[ClaimChannelSpec, ...]
    required_obligations: frozenset[str]
    static_providers: tuple[StaticProviderSpec, ...]
    max_input_tokens: int
    max_output_tokens: int
    max_attempts: int
    max_context_bytes: int
    maximum_cost_exposure: float
    attachment_policy: AttachmentPolicySpec = AttachmentPolicySpec()
    candidate_source_path: str | None = None
    temperature: float | None = None
    timeout: int | None = None
    reasoning_effort: str | None = None
    zero_cost_local: bool = False
    require_usage: bool = True
    supported: bool = True
    closure_blocker: str | None = None

    def channel(self, name: str) -> ClaimChannelSpec:
        """Return one exact channel policy or fail closed."""
        matches = [channel for channel in self.channels if channel.channel == name]
        if len(matches) != 1:
            raise DispatchPolicyRegistryError(
                f"Policy {self.policy_id!r} requires one {name!r} channel; "
                f"found {len(matches)}"
            )
        return matches[0]


@dataclass(frozen=True, slots=True)
class ResolvedDispatchPolicy:
    """Trusted policy plus the exact runtime objects named by its resource."""

    spec: DispatchPolicySpec
    model: str
    response_model: type[BaseModel]
    token_counter: Callable[[Any], int]


def _require_string(value: Any, location: str) -> str:
    if not isinstance(value, str) or not value.strip():
        raise DispatchPolicyRegistryError(f"{location} must be a non-blank string")
    return value


def _require_integer(value: Any, location: str, *, allow_zero: bool = False) -> int:
    if isinstance(value, bool) or not isinstance(value, int):
        raise DispatchPolicyRegistryError(f"{location} must be an integer")
    minimum = 0 if allow_zero else 1
    if value < minimum:
        raise DispatchPolicyRegistryError(f"{location} must be at least {minimum}")
    return value


def _string_set(value: Any, location: str) -> frozenset[str]:
    if not isinstance(value, list) or any(
        not isinstance(item, str) or not item.strip() for item in value
    ):
        raise DispatchPolicyRegistryError(f"{location} must be a string list")
    if len(value) != len(set(value)):
        raise DispatchPolicyRegistryError(f"{location} contains duplicates")
    return frozenset(value)


def _optional_number(value: Any, location: str) -> float | None:
    if value is None:
        return None
    if isinstance(value, bool) or not isinstance(value, int | float):
        raise DispatchPolicyRegistryError(f"{location} must be numeric")
    number = float(value)
    if not math.isfinite(number):
        raise DispatchPolicyRegistryError(f"{location} must be finite")
    return number


def _optional_string(value: Any, location: str) -> str | None:
    if value is None:
        return None
    return _require_string(value, location)


def _boolean(value: Any, location: str, *, default: bool) -> bool:
    if value is None:
        return default
    if not isinstance(value, bool):
        raise DispatchPolicyRegistryError(f"{location} must be a Boolean")
    return value


def _optional_timeout(value: Any, location: str) -> int | None:
    if value is None:
        return None
    return _require_integer(value, location)


def _load_spec(data: Mapping[str, Any], location: str) -> DispatchPolicySpec:
    templates_raw = data.get("templates")
    if not isinstance(templates_raw, list) or not templates_raw:
        raise DispatchPolicyRegistryError(f"{location}.templates must not be empty")
    templates: list[TemplateRoleSpec] = []
    for index, value in enumerate(templates_raw):
        if not isinstance(value, Mapping):
            raise DispatchPolicyRegistryError(
                f"{location}.templates[{index}] must be an object"
            )
        templates.append(
            TemplateRoleSpec(
                role=_require_string(value.get("role"), f"{location}.templates.role"),
                name=_require_string(value.get("name"), f"{location}.templates.name"),
                source_version=_require_string(
                    value.get("source_version"),
                    f"{location}.templates.source_version",
                ),
            )
        )
    roles = [template.role for template in templates]
    if roles[0] != "system" or any(role not in {"system", "user"} for role in roles):
        raise DispatchPolicyRegistryError(
            f"{location}.templates must use ordered system/user roles"
        )
    first_user = next(
        (index for index, role in enumerate(roles) if role == "user"), None
    )
    if first_user is not None and "system" in roles[first_user:]:
        raise DispatchPolicyRegistryError(
            f"{location}.templates cannot place system roles after user roles"
        )

    channels_raw = data.get("channels")
    if not isinstance(channels_raw, Mapping):
        raise DispatchPolicyRegistryError(f"{location}.channels must be an object")
    channels = tuple(
        ClaimChannelSpec(
            channel=_require_string(name, f"{location}.channels name"),
            kinds=_string_set(value.get("kinds"), f"{location}.channels.{name}.kinds"),
            scopes=_string_set(
                value.get("scopes"), f"{location}.channels.{name}.scopes"
            ),
        )
        for name, value in channels_raw.items()
        if isinstance(value, Mapping)
    )
    if len(channels) != len(channels_raw):
        raise DispatchPolicyRegistryError(
            f"{location}.channels values must all be objects"
        )

    providers_raw = data.get("static_providers", [])
    if not isinstance(providers_raw, list):
        raise DispatchPolicyRegistryError(f"{location}.static_providers must be a list")
    static_providers: list[StaticProviderSpec] = []
    for index, value in enumerate(providers_raw):
        if not isinstance(value, Mapping):
            raise DispatchPolicyRegistryError(
                f"{location}.static_providers[{index}] must be an object"
            )
        static_providers.append(
            StaticProviderSpec(
                name=_require_string(
                    value.get("name"), f"{location}.static_providers.name"
                ),
                kind=_require_string(
                    value.get("kind"), f"{location}.static_providers.kind"
                ),
                source_version=_require_string(
                    value.get("source_version"),
                    f"{location}.static_providers.source_version",
                ),
            )
        )

    attachments_raw = data.get("attachments", {})
    if not isinstance(attachments_raw, Mapping):
        raise DispatchPolicyRegistryError(f"{location}.attachments must be an object")
    attachment_policy = AttachmentPolicySpec(
        allowed_media_types=_string_set(
            attachments_raw.get("allowed_media_types", []),
            f"{location}.attachments.allowed_media_types",
        ),
        max_count=_require_integer(
            attachments_raw.get("max_count", 0),
            f"{location}.attachments.max_count",
            allow_zero=True,
        ),
        max_bytes_each=_require_integer(
            attachments_raw.get("max_bytes_each", 0),
            f"{location}.attachments.max_bytes_each",
            allow_zero=True,
        ),
        max_bytes_total=_require_integer(
            attachments_raw.get("max_bytes_total", 0),
            f"{location}.attachments.max_bytes_total",
            allow_zero=True,
        ),
        max_width=_require_integer(
            attachments_raw.get("max_width", 0),
            f"{location}.attachments.max_width",
            allow_zero=True,
        ),
        max_height=_require_integer(
            attachments_raw.get("max_height", 0),
            f"{location}.attachments.max_height",
            allow_zero=True,
        ),
    )
    if attachment_policy.max_count == 0:
        if (
            attachment_policy.allowed_media_types
            or attachment_policy.max_bytes_each
            or attachment_policy.max_bytes_total
            or attachment_policy.max_width
            or attachment_policy.max_height
        ):
            raise DispatchPolicyRegistryError(
                f"{location}.attachments must be empty when max_count is zero"
            )
    elif (
        not attachment_policy.allowed_media_types
        or attachment_policy.max_bytes_each <= 0
        or attachment_policy.max_bytes_total <= 0
        or attachment_policy.max_width <= 0
        or attachment_policy.max_height <= 0
    ):
        raise DispatchPolicyRegistryError(
            f"{location}.attachments requires media and positive byte/dimension bounds"
        )
    maximum_cost = _optional_number(
        data.get("maximum_cost_exposure"), f"{location}.maximum_cost_exposure"
    )
    if maximum_cost is None or maximum_cost < 0:
        raise DispatchPolicyRegistryError(
            f"{location}.maximum_cost_exposure must be non-negative"
        )
    identifier_pattern = _require_string(
        data.get("identifier_pattern"), f"{location}.identifier_pattern"
    )
    try:
        re.compile(identifier_pattern)
    except re.error as exc:
        raise DispatchPolicyRegistryError(
            f"{location}.identifier_pattern is invalid"
        ) from exc
    return DispatchPolicySpec(
        policy_id=_require_string(data.get("policy_id"), f"{location}.policy_id"),
        source_version=_require_string(
            data.get("source_version"), f"{location}.source_version"
        ),
        callsite_id=_require_string(data.get("callsite_id"), f"{location}.callsite_id"),
        service=_require_string(data.get("service"), f"{location}.service"),
        seat=_require_string(data.get("seat"), f"{location}.seat"),
        task_kind=_require_string(data.get("task_kind"), f"{location}.task_kind"),
        templates=tuple(templates),
        response_model_path=_require_string(
            data.get("response_model"), f"{location}.response_model"
        ),
        model_section=_require_string(
            data.get("model_section"), f"{location}.model_section"
        ),
        tokenizer_path=_require_string(data.get("tokenizer"), f"{location}.tokenizer"),
        tokenizer_key=_require_string(
            data.get("tokenizer_key"), f"{location}.tokenizer_key"
        ),
        identifier_pattern=identifier_pattern,
        channels=channels,
        required_obligations=_string_set(
            data.get("required_obligations", []),
            f"{location}.required_obligations",
        ),
        static_providers=tuple(static_providers),
        max_input_tokens=_require_integer(
            data.get("max_input_tokens"), f"{location}.max_input_tokens"
        ),
        max_output_tokens=_require_integer(
            data.get("max_output_tokens"), f"{location}.max_output_tokens"
        ),
        max_attempts=_require_integer(
            data.get("max_attempts"), f"{location}.max_attempts"
        ),
        max_context_bytes=_require_integer(
            data.get("max_context_bytes"), f"{location}.max_context_bytes"
        ),
        maximum_cost_exposure=maximum_cost,
        attachment_policy=attachment_policy,
        candidate_source_path=_optional_string(
            data.get("candidate_source"), f"{location}.candidate_source"
        ),
        temperature=_optional_number(
            data.get("temperature"), f"{location}.temperature"
        ),
        timeout=_optional_timeout(data.get("timeout"), f"{location}.timeout"),
        reasoning_effort=_optional_string(
            data.get("reasoning_effort"), f"{location}.reasoning_effort"
        ),
        zero_cost_local=_boolean(
            data.get("zero_cost_local"), f"{location}.zero_cost_local", default=False
        ),
        require_usage=_boolean(
            data.get("require_usage"), f"{location}.require_usage", default=True
        ),
        supported=_boolean(
            data.get("supported"), f"{location}.supported", default=True
        ),
        closure_blocker=_optional_string(
            data.get("closure_blocker"), f"{location}.closure_blocker"
        ),
    )


def load_dispatch_policy_registry(
    directory: Path = DISPATCH_POLICY_DIR,
) -> Mapping[str, DispatchPolicySpec]:
    """Load every domain-owned JSON resource into one immutable exact registry."""
    policies: dict[str, DispatchPolicySpec] = {}
    if not directory.exists():
        return MappingProxyType(policies)
    for path in sorted(directory.glob("*.json")):
        if path.name.endswith(".blocked.json"):
            continue
        try:
            payload = json.loads(path.read_text())
        except (OSError, json.JSONDecodeError) as exc:
            raise DispatchPolicyRegistryError(
                f"Cannot load dispatch policy resource {path}"
            ) from exc
        entries = payload.get("policies") if isinstance(payload, Mapping) else None
        if not isinstance(entries, list):
            raise DispatchPolicyRegistryError(f"{path} must contain a policies list")
        for index, entry in enumerate(entries):
            if not isinstance(entry, Mapping):
                raise DispatchPolicyRegistryError(
                    f"{path}:policies[{index}] is invalid"
                )
            spec = _load_spec(entry, f"{path}:policies[{index}]")
            if spec.callsite_id in policies:
                raise DispatchPolicyRegistryError(
                    f"Duplicate typed dispatch policy for {spec.callsite_id!r}"
                )
            get_callsite_registration(spec.callsite_id)
            policies[spec.callsite_id] = spec
    return MappingProxyType(policies)


def load_dispatch_closure_blockers(
    directory: Path = DISPATCH_POLICY_DIR,
) -> Mapping[str, str]:
    """Load explicit typed-route blockers that prevent false migration closure."""
    blockers: dict[str, str] = {}
    if not directory.exists():
        return MappingProxyType(blockers)
    for path in sorted(directory.glob("*.blocked.json")):
        try:
            payload = json.loads(path.read_text())
        except (OSError, json.JSONDecodeError) as exc:
            raise DispatchPolicyRegistryError(
                f"Cannot load typed dispatch blockers {path}"
            ) from exc
        entries = payload.get("unsupported") if isinstance(payload, Mapping) else None
        if not isinstance(entries, list):
            raise DispatchPolicyRegistryError(
                f"{path} must contain an unsupported list"
            )
        for index, entry in enumerate(entries):
            if not isinstance(entry, Mapping):
                raise DispatchPolicyRegistryError(
                    f"{path}:unsupported[{index}] must be an object"
                )
            callsite_id = _require_string(
                entry.get("callsite_id"), f"{path}:unsupported.callsite_id"
            )
            reason = _require_string(
                entry.get("closure_blocker"), f"{path}:unsupported.closure_blocker"
            )
            if callsite_id in blockers:
                raise DispatchPolicyRegistryError(
                    f"Duplicate typed closure blocker for {callsite_id!r}"
                )
            get_callsite_registration(callsite_id)
            blockers[callsite_id] = reason
    return MappingProxyType(blockers)


def _import_object(path: str) -> Any:
    module_name, separator, qualname = path.partition(":")
    if not separator or not module_name or not qualname:
        raise DispatchPolicyRegistryError(
            f"Registered object path must use module:qualname syntax: {path!r}"
        )
    try:
        value: Any = importlib.import_module(module_name)
        for part in qualname.split("."):
            value = getattr(value, part)
    except (ImportError, AttributeError) as exc:
        raise DispatchPolicyRegistryError(
            f"Cannot resolve registered object {path!r}"
        ) from exc
    return value


def resolve_dispatch_policy(
    callsite_id: str,
    *,
    candidate_model: str | None = None,
    registry: Mapping[str, DispatchPolicySpec] | None = None,
) -> ResolvedDispatchPolicy:
    """Resolve one trusted policy and only its explicitly permitted model axis."""
    from imas_codex.settings import get_model

    active = DISPATCH_POLICY_REGISTRY if registry is None else registry
    spec = active.get(callsite_id)
    if spec is None:
        blocker = DISPATCH_CLOSURE_BLOCKERS.get(callsite_id)
        if blocker:
            raise DispatchPolicyRegistryError(
                f"Callsite {callsite_id!r} is typed-unsupported: {blocker}"
            )
        raise DispatchPolicyRegistryError(
            f"Callsite {callsite_id!r} has no typed-ready dispatch policy"
        )
    if not spec.supported:
        reason = spec.closure_blocker or "route is explicitly unsupported"
        raise DispatchPolicyRegistryError(f"Callsite {callsite_id!r}: {reason}")
    configured_model = get_model(spec.model_section)
    model = configured_model
    if candidate_model is not None:
        if spec.candidate_source_path is None:
            raise DispatchPolicyRegistryError(
                f"Policy {spec.policy_id!r} does not permit a candidate model"
            )
        source = _import_object(spec.candidate_source_path)
        if not callable(source):
            raise DispatchPolicyRegistryError(
                f"Candidate source {spec.candidate_source_path!r} is not callable"
            )
        candidates = tuple(source())
        if candidate_model not in candidates:
            raise DispatchPolicyRegistryError(
                f"Candidate model is outside {spec.candidate_source_path!r}"
            )
        model = candidate_model
    response_model = _import_object(spec.response_model_path)
    if not isinstance(response_model, type) or not issubclass(
        response_model, BaseModel
    ):
        raise DispatchPolicyRegistryError(
            f"Response model {spec.response_model_path!r} is not a Pydantic model"
        )
    token_counter = _import_object(spec.tokenizer_path)
    if not callable(token_counter):
        raise DispatchPolicyRegistryError(
            f"Tokenizer {spec.tokenizer_path!r} is not callable"
        )
    return ResolvedDispatchPolicy(spec, model, response_model, token_counter)


def policy_registry_closure(
    observed_calls: Iterable[Any],
    *,
    registry: Mapping[str, DispatchPolicySpec] | None = None,
) -> tuple[int, int]:
    """Return legacy/typed counts and require every typed call to have a policy."""
    active = DISPATCH_POLICY_REGISTRY if registry is None else registry
    legacy = 0
    typed = 0
    for call in observed_calls:
        if getattr(call, "transition_kind", "legacy") == "typed":
            typed += 1
            callsite_id = getattr(call, "callsite_id", None)
            if callsite_id not in active:
                raise DispatchPolicyRegistryError(
                    f"Typed expression has no policy: {callsite_id!r}"
                )
        else:
            legacy += 1
    return legacy, typed


DISPATCH_POLICY_REGISTRY = load_dispatch_policy_registry()
DISPATCH_CLOSURE_BLOCKERS = load_dispatch_closure_blockers()


__all__ = [
    "AttachmentPolicySpec",
    "ClaimChannelSpec",
    "DISPATCH_POLICY_DIR",
    "DISPATCH_POLICY_REGISTRY",
    "DISPATCH_CLOSURE_BLOCKERS",
    "DispatchPolicyRegistryError",
    "DispatchPolicySpec",
    "ResolvedDispatchPolicy",
    "StaticProviderSpec",
    "TemplateRoleSpec",
    "load_dispatch_policy_registry",
    "load_dispatch_closure_blockers",
    "policy_registry_closure",
    "resolve_dispatch_policy",
]
